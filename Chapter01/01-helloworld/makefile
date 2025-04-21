# 定义 CUDA 安装路径
CUDA_PATH=/usr/local/cuda
# 定义主机编译器，默认使用 g++
HOST_COMPILER ?= g++
# 定义 NVCC 编译器，指定使用的主机编译器
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
# 定义目标可执行文件的名称
TARGET=hello_world vector_add

# 定义包含路径，包含 CUDA 示例的公共头文件目录
INCLUDES= -I${CUDA_PATH}/samples/common/inc
# 定义 NVCC 编译选项，使用 64 位模式并显示行号信息
NVCC_FLAGS=-m64 -lineinfo

# 检查 CUDA 版本是否大于等于 11.0
IS_CUDA_11:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode 参数，指定支持的 CUDA 架构
SMS = 35 37 50 52 60 61 70 75
# 如果 CUDA 版本大于等于 11.0，更新支持的架构
ifeq "$(IS_CUDA_11)" "1"
SMS = 52 60 61 70 75 80
endif
# 遍历每个架构，生成对应的 gencode 标志
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# 编译 hello_world 目标，依赖于 hello_world.cu 文件
hello_world: hello_world.cu
	${NVCC} ${INCLUDES} ${NVCC_FLAGS} ${GENCODE_FLAGS} -o $@ $<

# 编译 vector_add 目标，依赖于 vector_add.cu 文件
vector_add: vector_add.cu
	${NVCC} ${INCLUDES} ${NVCC_FLAGS} ${GENCODE_FLAGS} -o $@ $<

matrix_mul: matrix_mul.cu
	${NVCC} ${INCLUDES} ${NVCC_FLAGS} ${GENCODE_FLAGS} -o $@ $<

# 清理规则，删除所有目标可执行文件
clean:
	rm -f ${TARGET}