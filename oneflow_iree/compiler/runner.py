from oneflow.nn.graph import Graph
import oneflow as flow
from google.protobuf import text_format
from iree import runtime as ireert
from iree.compiler import compile_str
import numpy as np
import time


class Backend:
    class NotFoundError(RuntimeError):
        def __init__(self, arg):
            self.args = arg

    class NotSupportError(RuntimeError):
        def __init__(self, arg):
            self.args = arg

    enable_tick: bool

    def __init__(self, enable_tick=True):
        self.enable_tick = enable_tick

    def tick(self, msg="", reset=False):
        if self.enable_tick == True:
            if reset == True:
                self.timer = time.time()
                if msg != "" and self.start_timer > 0:
                    current = time.time()
                    gap = current - self.timer
                    self.timer = current
                    print(msg + ": ")
                    print("  - stage cost " + str(gap) + 's')
                    print("  - total cost " + str(current) + 's')
                else:
                    print("restart tick")
                self.start_timer = self.timer
            else:
                current = time.time()
                gap = current - self.timer
                self.timer = current
                print(msg + ": ")
                print("  - stage cost " + str(gap) + 's')
                print("  - total cost " + str(current) + 's')


class Iree(Backend):
    class Target:
        pass

    class Cpu(Target):
        backend = ["dylib-llvm-aot"]
        config = "dylib"

    class Cuda(Target):
        backend = ["cuda"]
        config = "cuda"

    # members
    graph: Graph
    job: str
    tosa: str
    ctx: ireert.SystemContext

    def __init__(self, target=Cpu, enable_tick=True):
        super().__init__(enable_tick)
        self.target = target

    def cpu(self):
        self.target = Iree.Cpu

    def cuda(self):
        self.target = Iree.Cuda

    def generate_vm_module(self, graph: Graph):
        self.graph = graph
        self._get_job()
        self._convert_job_to_tosa()
        self.tick("convert job to tosa")
        self._convert_tosa_to_flat_buffer()
        self.tick("compile tosa to iree bytecode")
        self._convert_flat_buffer_to_vm_module()
        self.tick("compile iree bytecode to vm module")


    def _convert_tosa_to_flat_buffer(self):
        self.flat_buffer = compile_str(
            self.tosa, target_backends=self.target.backend, input_type="tosa"
        )

    def _convert_flat_buffer_to_vm_module(self):
        self.vm_module = ireert.VmModule.from_flatbuffer(self.flat_buffer)

    def _get_job(self):
        self.job = str(text_format.MessageToString(self.graph._forward_job_proto))

    def _convert_job_to_tosa(self):
        self.tosa = flow._oneflow_internal.nn.graph.ConvertJobToTosaIR(self.job)

    def generate_context(self):
        config = ireert.Config(self.target.config)
        self.ctx = ireert.SystemContext(config=config)
        self.ctx.add_vm_module(self.vm_module)
        self.tick("create iree vm context")
        return self.ctx


class Runner(object):

    _tosa_cache = {}

    def __init__(self, raw_graph, backend=Iree, return_numpy=True):
        self.raw_graph = raw_graph
        if backend == Iree or backend == "iree":
            self.backend = backend()
            if not return_numpy:
                raise Backend.NotSupportError("iree backend only supports return numpy")
        else:
            raise Backend.NotFoundError(str(backend) + "not found")
        self.return_numpy = return_numpy

    def cuda(self):
        self.backend.cuda()
        return self

    def cpu(self):
        self.backend.cpu()
        return self

    def _parse_input(self, *args, **kwargs):
        res = []
        for arg in args:
            if isinstance(arg, flow.Tensor):
                res.append(arg.cpu().detach().numpy())
            elif isinstance(arg, np.ndarray):
                res.append(arg)
            else:
                print("not support class")
                exit(1)
        return res

    def _get_function(self, *args, **kwargs):
        full_name = self._full_name()
        if not full_name in Runner._tosa_cache:
            graph = self.raw_graph()
            # graph.build_graph(*args, **kwargs)
            graph._compile(*args, **kwargs)
            self.backend.generate_vm_module(graph)
            Runner._tosa_cache[full_name] = {
                "name": graph._name,
                "data": self.backend.vm_module,
            }
        config = Runner._tosa_cache[full_name]
        self.backend.vm_module = config["data"]
        ctx = self.backend.generate_context()
        f = ctx.modules.module[config["name"]]
        return f

    def _parse_output(self, output):
        if output.is_host_accessible:
            return output
        else:
            return output.to_host()

    def _full_name(self):
        full_name = self.raw_graph.__name__
        for elem in self.input:
            full_name += str(elem.shape) + str(elem.dtype) + str(self.backend.target)
        return full_name

    def __call__(self, *args, **kwargs):
        self.backend.tick(reset=True)
        self.input = self._parse_input(*args, **kwargs)
        function = self._get_function(*args, **kwargs)
        output = function(*self.input)
        self.backend.tick("run module")
        return self._parse_output(output)
