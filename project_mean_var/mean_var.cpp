#include "mean_var.hpp"

void DEBUG()
{
    std::cout<<"HELLO"<<std::endl;
}

int main()
{
    try
    {
        // Get Queue, Device, Context, Platform
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Platform platform{device.getInfo<CL_DEVICE_PLATFORM>()};

        // Load reduction.cl kernel source file
        std::ifstream source_file{ "../reduction.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "reduction.cl" };
    
        // Create cl::Program from kernel and build it for the device
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} } };
        program.build({ device });

        // Create a dummy testing vector
        size_t N = 2048;
        std::vector<float> data(N);
        for (int i = 0; i < N; ++i)
            data[i] = 1.0;
        
        // Create kernel from program
        cl::Kernel kernel(program, "reduction");

        // Access work group size 
        auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        auto numWorkGroups = data.size() / workGroupSize;

        // Result vector holding the partial sums
        std::vector<float> result(numWorkGroups);

        // Create buffers
        cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * data.size(), data.data());
        cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

        // reduction kernel takes 3 arguments
        kernel.setArg(0, buf_in);                                 //__global float* data
        kernel.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
        kernel.setArg(2, buf_out);                                //__global float* result


        // Start kernel and read the buf_out
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
        queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, result.data());

        // Print stuff
        std::cout << "data size (N) = " << N << std::endl;
        std::cout << "work group size = " << workGroupSize << std::endl;
        std::cout << "number of work groups = " << numWorkGroups << std::endl;
        std::cout << "Results: " << std::endl;
        for(int i = 0; i < numWorkGroups; ++i)
            std::cout << result[i] << std::endl;

    }/// end of try case
    
    catch (cl::BuildError& error) // If kernel failed to build
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        for (const auto& log : error.getBuildLog())
            std::cerr << "\tBuild log for device: " << log.first.getInfo<CL_DEVICE_NAME>() << std::endl << std::endl << log.second << std::endl << std::endl;
        std::exit(error.err());
    }
    catch (cl::Error& error) // If any OpenCL error occurs
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        std::exit(error.err());
    }
    catch (std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
