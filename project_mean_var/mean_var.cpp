#include "mean_var.hpp"

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
        size_t N_original = 1000; // N should be between 2 - 2^24 for the project
        size_t N = N_original;    // will be overwritten at each kernel launches
        std::vector<float> data(N);
        for (int i = 0; i < N; ++i)
            data[i] = 1.0;
        
         // Create kernel from program
        cl::Kernel kernel(program, "reduction");

        // Access work group size 
        auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        
        // Check if N % workGroupSize is 0, if not, append 0s to data
        if (N % workGroupSize != 0)
        {
            data.resize(N + (workGroupSize - (N % workGroupSize)), 0.0);
            std::cout << "Resize was needed, from N = " << N << " to N = " << data.size() << " (appended elements are zeros)!" <<std::endl;
            N = data.size();
        }
        
        // Determine how many kernel launches will be needed (can be 1 ,2 ,3 based on N)
        int n_launch = 1;
        bool enough_launches = false;
        int N_extended = data.size();
        while(not enough_launches)
        {
            if(N_extended / workGroupSize != 0)
            {
                N_extended /= workGroupSize;
                n_launch += 1;
            }
            else
                enough_launches = true;
        }
        std::cout << "Number of required kernel launches = "<< n_launch <<" (for N = " << N << ", work group size = " << workGroupSize << ")"<< std::endl;

        // Vectors for storing the results for each kernel launched
        std::vector<float> result1, result2, result3;
        
        // The first kernel launch will always be needed
        if(true)
        {
            // Number of work groups
            auto numWorkGroups = data.size() / workGroupSize;
            
            // Result vector holding the first kernel launch partial sums
            result1.resize(numWorkGroups);

            // Create buffers
            cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * data.size(), data.data());
            cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

            // reduction kernel takes 3 arguments
            kernel.setArg(0, buf_in);                                 //__global float* data
            kernel.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
            kernel.setArg(2, buf_out);                                //__global float* result


            // Start kernel and read the buf_out
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
            queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, result1.data());
        }

        std::cout<<"result1 vector"<<std::endl;
        for(int i=0;i<result1.size();++i)
            std::cout<<result1[i]<<std::endl;

        // Second kernel launch
        if(result1.size() > 1)
        {
            // Check if N % workGroupSize is 0, if not, append 0s to data
            N = result1.size();
            if (N % workGroupSize != 0)
            {
                result1.resize(N + (workGroupSize - (N % workGroupSize)), 0.0);
                std::cout << "Resize was needed, from N = " << N << " to N = " << result1.size() << " (appended elements are zeros)!" <<std::endl;
                N = result1.size();
            }

            // Number of work groups
            auto numWorkGroups = result1.size() / workGroupSize;
            
            // Result vector holding the second kernel launch partial sums
            result2.resize(numWorkGroups);

            // Create buffers
            cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * result1.size(), result1.data());
            cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

            // reduction kernel takes 3 arguments
            kernel.setArg(0, buf_in);                                 //__global float* data
            kernel.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
            kernel.setArg(2, buf_out);                                //__global float* result


            // Start kernel and read the buf_out
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
            queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, result2.data());
        }

        std::cout<<"result2 vector"<<std::endl;
        for(int i=0;i<result2.size();++i)
            std::cout<<result2[i]<<std::endl;

        // Third kernel launch
        if(result2.size() > 1)
        {
            // Check if N % workGroupSize is 0, if not, append 0s to data
            N = result2.size();
            if (N % workGroupSize != 0)
            {
                result2.resize(N + (workGroupSize - (N % workGroupSize)), 0.0);
                std::cout << "Resize was needed, from N = " << N << " to N = " << result2.size() << " (appended elements are zeros)!" <<std::endl;
                N = result2.size();
            }

            // Number of work groups
            auto numWorkGroups = result2.size() / workGroupSize;
            
            // Result vector holding the third kernel launch partial sums
            result3.resize(numWorkGroups);

            // Create buffers
            cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * result2.size(), result2.data());
            cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

            // reduction kernel takes 3 arguments
            kernel.setArg(0, buf_in);                                 //__global float* data
            kernel.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
            kernel.setArg(2, buf_out);                                //__global float* result


            // Start kernel and read the buf_out
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
            queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, result3.data());
        }

        std::cout<<"result3 vector"<<std::endl;
        for(int i=0;i<result3.size();++i)
            std::cout<<result3[i]<<std::endl;
       

        



        /*
        // Check if N % workGroupSize is 0, if not, append 0s to data
        if (N % workGroupSize != 0)
        {
            data.resize(N + (workGroupSize - (N % workGroupSize)), 0.0);
            std::cout << "Resize was needed, from N = " << N << " to N = " << data.size() << " (appended elements are zeros)!" <<std::endl;
            N = data.size();
        }

        // Number of work groups
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
        */
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
