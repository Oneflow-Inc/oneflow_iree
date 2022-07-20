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

    def __init__(self, target=Cpu):
        super().__init__()
        self.target = target

    def cpu(self):
        self.target = Iree.Cpu

    def cuda(self):
        self.target = Iree.Cuda

    def generate_vm_module(self, graph: Graph):
        self.graph = graph
        self._get_job()
        self._convert_job_to_tosa()
        self._convert_tosa_to_flat_buffer()
        self._convert_flat_buffer_to_vm_module()


    def _convert_tosa_to_flat_buffer(self):
        self.flat_buffer = compile_str(
            self.tosa, target_backends=self.target.backend, input_type="tosa"
        )

    def _convert_flat_buffer_to_vm_module(self):
        self.vm_module = ireert.VmModule.from_flatbuffer(self.flat_buffer)

    def _get_job(self):
        self.job = self.graph._full_job_proto.SerializeToString()


    def _convert_job_to_tosa(self):
        self.tosa = flow._oneflow_internal.nn.graph.ConvertJobToTosaIR(self.job)

    def generate_context(self):
        config = ireert.Config(self.target.config)
        self.ctx = ireert.SystemContext(config=config)
        self.ctx.add_vm_module(self.vm_module)
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
        self.input = self._parse_input(*args, **kwargs)
        function = self._get_function(*args, **kwargs)
        output = function(*self.input)
        return self._parse_output(output)
