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
        size_t N_original = 2500; // N should be between 2 - 2^24 = 16777216 for the project
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
            std::cout << "starting data: resize was needed, from N = " << N << " to N = " << data.size() << " (appended elements are zeros)!" <<std::endl;
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
        std::vector< std::vector<float> > results(n_launch);

        for(int ilaunch = 0; ilaunch < n_launch; ++ilaunch)
        {
            int numWorkGroups;

            // First kernel launch always needed
            if (ilaunch == 0)
            {
                numWorkGroups = data.size() / workGroupSize;
                
                // Result vector holding the first kernel launch partial sums
                results[ilaunch].resize(numWorkGroups);

                // Create buffers
                cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * data.size(), data.data());
                cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

                // reduction kernel takes 3 arguments
                kernel.setArg(0, buf_in);                                 //__global float* data
                kernel.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
                kernel.setArg(2, buf_out);                                //__global float* result


                // Start kernel and read the buf_out
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
                queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results[ilaunch].data());
            }

            // If more kernel launches are needed
            else
            {
                // Check if N % workGroupSize is 0, if not, append 0s to data
                N = results[ilaunch - 1].size();
                if (N % workGroupSize != 0)
                {
                    results[ilaunch - 1].resize(N + (workGroupSize - (N % workGroupSize)), 0.0);
                    std::cout << "ilaunch" <<ilaunch << ": resize was needed, from N = " << N << " to N = " << results[ilaunch - 1].size() << " (appended elements are zeros)!" <<std::endl;
                    N = results[ilaunch - 1].size();
                }
                // Number of work groups
                numWorkGroups = results[ilaunch - 1].size() / workGroupSize;

                // Result vector holding the (ilaunch+1)th kernel launch partial sums
                results[ilaunch].resize(numWorkGroups);

                // Create buffers
                cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * results[ilaunch - 1].size(), results[ilaunch - 1].data());
                cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

                // reduction kernel takes 3 arguments
                kernel.setArg(0, buf_in);                                 //__global float* data
                kernel.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
                kernel.setArg(2, buf_out);                                //__global float* result


                // Start kernel and read the buf_out
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
                queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results[ilaunch].data());

            }
            
        }

        // Check if final result vector indeed stores 1 final value
        std::cout << "number of elements in last result vector = " << results[n_launch - 1].size() << std::endl;
        std::cout << "Final result is: " << results[n_launch - 1][0] << std::endl;

       
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
