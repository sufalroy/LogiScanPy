import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
from cuda import cudart
from numpy import ndarray

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


@dataclass
class Tensor:
    """
    A class to represent a tensor.

    Attributes:
        name (str): The name of the tensor.
        dtype (np.dtype): The data type of the tensor.
        shape (Tuple): The shape of the tensor.
        cpu (ndarray): The CPU data of the tensor.
        gpu (int): The GPU memory pointer of the tensor.
    """
    name: str
    dtype: np.dtype
    shape: Tuple
    cpu: ndarray
    gpu: int


class TensorRTEngine:
    """
    A class to represent a TensorRT engine.

    Attributes:
        weight (Path): The path to the TensorRT engine file.
        stream (int): The CUDA stream handle.
        num_bindings (int): The number of bindings in the engine.
        bindings (List[int]): A list of GPU memory pointers for the bindings.
        num_inputs (int): The number of input bindings.
        num_outputs (int): The number of output bindings.
        model (trt.ICudaEngine): The TensorRT engine model.
        context (trt.IExecutionContext): The execution context of the engine.
        input_names (List[str]): A list of names of the input bindings.
        output_names (List[str]): A list of names of the output bindings.
        is_dynamic (bool): A flag indicating if the engine has dynamic axes.
        inp_info (List[Tensor]): A list of input tensor information.
        out_info (List[Tensor]): A list of output tensor information.
        out_ptrs (List[int]): A list of GPU memory pointers for the output bindings.
    """

    def __init__(self, weight: Union[str, Path]) -> None:
        """
        Initializes the TensorRTEngine.

        Args:
            weight (Union[str, Path]): The path to the TensorRT engine file.
        """
        self.weight = Path(weight) if isinstance(weight, str) else weight
        status, self.stream = cudart.cudaStreamCreate()
        assert status.value == 0
        self.__init_engine()
        self.__init_bindings()
        self.__warm_up()

    def __init_engine(self) -> None:
        """
        Initializes the TensorRT engine and execution context.
        """
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()

        names = [model.get_binding_name(i) for i in range(model.num_bindings)]
        self.num_bindings = model.num_bindings
        self.bindings: List[int] = [0] * self.num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(model.num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]

    def __init_bindings(self) -> None:
        """
        Initializes the bindings for the TensorRT engine.
        """
        dynamic = False
        inp_info = []
        out_info = []
        out_ptrs = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                dynamic |= True
            if not dynamic:
                cpu = np.empty(shape, dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            else:
                cpu, gpu = np.empty(0), 0
            inp_info.append(Tensor(name, dtype, shape, cpu, gpu))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if not dynamic:
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
                out_ptrs.append(gpu)
            else:
                cpu, gpu = np.empty(0), 0
            out_info.append(Tensor(name, dtype, shape, cpu, gpu))

        self.is_dynamic = dynamic
        self.inp_info = inp_info
        self.out_info = out_info
        self.out_ptrs = out_ptrs

    def __warm_up(self) -> None:
        """
        Performs a warm-up inference on the TensorRT engine.
        """
        if self.is_dynamic:
            print('You engine has dynamic axes, please warm up by yourself !')
            return
        for _ in range(10):
            inputs = []
            for i in self.inp_info:
                inputs.append(i.cpu)
            self.__call__(inputs)

    def set_profiler(self, profiler: Optional[trt.IProfiler]) -> None:
        """
        Sets the profiler for the TensorRT engine.

        Args:
            profiler (Optional[trt.IProfiler]): The profiler to be set.
        """
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def __call__(self, *inputs) -> Union[Tuple, ndarray]:
        """
        Runs inference on the TensorRT engine.

        This method first checks if the number of inputs matches the expected number of inputs for the engine.
        It then creates a list of contiguous input arrays, which is required for efficient memory transfer.

        For each input, if the engine has dynamic axes, the method sets the binding shape for the input and
        allocates GPU memory for the input data. It then copies the input data from CPU to GPU using
        cudaMemcpyAsync.

        Next, the method allocates GPU memory for the outputs if the engine has dynamic axes. It sets the
        output pointers in the bindings list and executes the engine asynchronously using execute_async_v2.
        After synchronizing the CUDA stream, it copies the output data from GPU to CPU using cudaMemcpyAsync.

        Finally, it returns the output data as a tuple if there are multiple outputs, or a single numpy array
        if there is only one output.

        Args:
            *inputs (Tuple[ndarray]): The input data for the engine.

        Returns:
            Union[Tuple, ndarray]: The output data from the engine.
        """
        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[ndarray] = [
            np.ascontiguousarray(i) for i in inputs
        ]

        for i in range(self.num_inputs):

            if self.is_dynamic:
                self.context.set_binding_shape(
                    i, tuple(contiguous_inputs[i].shape))
                status, self.inp_info[i].gpu = cudart.cudaMallocAsync(
                    contiguous_inputs[i].nbytes, self.stream)
                assert status.value == 0
            cudart.cudaMemcpyAsync(
                self.inp_info[i].gpu, contiguous_inputs[i].ctypes.data,
                contiguous_inputs[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            self.bindings[i] = self.inp_info[i].gpu

        output_gpu_ptrs: List[int] = []
        outputs: List[ndarray] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                dtype = self.out_info[i].dtype
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            else:
                cpu = self.out_info[i].cpu
                gpu = self.out_info[i].gpu
            outputs.append(cpu)
            output_gpu_ptrs.append(gpu)
            self.bindings[j] = gpu

        self.context.execute_async_v2(self.bindings, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        for i, o in enumerate(output_gpu_ptrs):
            cudart.cudaMemcpyAsync(
                outputs[i].ctypes.data, o, outputs[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]
