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

        // Load kernel source files
        std::ifstream source_file{ "../reduction.cl" }; // kernel performing reduction
        std::ifstream source_file_mean{ "../mean.cl" }; // kernel performing reduction then dividing with given N
        std::ifstream source_file_var_reduction{ "../var_reduction.cl" }; // kernel performing reduction of (x_i - mean)**2
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "reduction.cl" };
        if (!source_file_mean.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "mean.cl" };
        if (!source_file_var_reduction.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "var_reduction.cl" };
    
        // Create cl::Program from kernels and build them for the device
        cl::Program program_reduction{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                        std::istreambuf_iterator<char>{} } };
        cl::Program program_mean{ std::string{ std::istreambuf_iterator<char>{ source_file_mean },
                                        std::istreambuf_iterator<char>{} } };
        cl::Program program_var_reduction{ std::string{ std::istreambuf_iterator<char>{ source_file_var_reduction },
                                        std::istreambuf_iterator<char>{} } };
        program_reduction.build({ device });
        program_mean.build({ device });
        program_var_reduction.build({ device});

        // Create a dummy testing vector
        size_t N_original = 1024; // N should be between 2 - 2^24 (=16777216) for the project
        size_t N = N_original;    // will be overwritten at each kernel launches
        std::vector<float> data(N);
        for (int i = 0; i < N; ++i)
            data[i] = 1.0;
        
         // Create kernels from programs
        cl::Kernel kernel_reduction(program_reduction, "reduction");
        cl::Kernel kernel_mean(program_mean, "mean");
        cl::Kernel kernel_var_reduction(program_var_reduction, "var_reduction");

        // Access work group size 
        auto workGroupSize = kernel_reduction.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        
        // Check if N % workGroupSize is 0, if not, append 0s to data
        if (N % workGroupSize != 0)
        {
            data.resize(N + (workGroupSize - (N % workGroupSize)), 0.0);
            std::cout << "LOG: starting data, resize was needed, from N = " << N << " to N = " << data.size() << " (appended elements are zeros)!" <<std::endl;
            N = data.size();
        }
        
        // Determine how many kernel launches will be needed (based on N)
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
        std::cout << "LOG: number of required kernel launches = "<< n_launch <<" (for N = " << N << ", work group size = " << workGroupSize << ")"<< std::endl;

        // Vectors for storing the results for each kernel launched
        std::vector< std::vector<float> > results_mean(n_launch);
        std::vector< std::vector<float> > results_var(n_launch);

        for(int ilaunch = 0; ilaunch < n_launch; ++ilaunch)
        {
            int numWorkGroups;
            // First kernel launch always needed
            if (ilaunch == 0)
            {
                numWorkGroups = data.size() / workGroupSize;
                
                // Result vector holding the first kernel launch partial sums
                results_mean[ilaunch].resize(numWorkGroups);

                // Create buffers
                cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * data.size(), data.data());
                cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

                // reduction kernel takes 3 arguments
                kernel_reduction.setArg(0, buf_in);                                 //__global float* data
                kernel_reduction.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
                kernel_reduction.setArg(2, buf_out);                                //__global float* result


                // Start kernel and read the buf_out
                queue.enqueueNDRangeKernel(kernel_reduction, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
                queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_mean[ilaunch].data());
            }
            // If more kernel launches are needed but not the last kernel launch
            else if ((ilaunch > 0) && (ilaunch < (n_launch - 1)))
            {
                // Check if N % workGroupSize is 0, if not, append 0s to data
                N = results_mean[ilaunch - 1].size();
                if (N % workGroupSize != 0)
                {
                    results_mean[ilaunch - 1].resize(N + (workGroupSize - (N % workGroupSize)), 0.0);
                    std::cout << "LOG: mean computation ilaunch" <<ilaunch << ": resize was needed, from N = " << N << " to N = " << results_mean[ilaunch - 1].size() << " (appended elements are zeros)!" <<std::endl;
                    N = results_mean[ilaunch - 1].size();
                }
                // Number of work groups
                numWorkGroups = results_mean[ilaunch - 1].size() / workGroupSize;

                // Result vector holding the (ilaunch+1)th kernel launch partial sums
                results_mean[ilaunch].resize(numWorkGroups);

                // Create buffers
                cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * results_mean[ilaunch - 1].size(), results_mean[ilaunch - 1].data());
                cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

                // reduction kernel takes 3 arguments
                kernel_reduction.setArg(0, buf_in);                                 //__global float* data
                kernel_reduction.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
                kernel_reduction.setArg(2, buf_out);                                //__global float* result


                // Start kernel and read the buf_out
                queue.enqueueNDRangeKernel(kernel_reduction, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
                queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_mean[ilaunch].data());

            }
            // Last kernel launch
            else if (ilaunch == (n_launch - 1))
            {
                // Check if N % workGroupSize is 0, if not, append 0s to data
                N = results_mean[ilaunch - 1].size();
                if (N % workGroupSize != 0)
                {
                    results_mean[ilaunch - 1].resize(N + (workGroupSize - (N % workGroupSize)), 0.0);
                    std::cout << "LOG: mean computation ilaunch" <<ilaunch << ": resize was needed, from N = " << N << " to N = " << results_mean[ilaunch - 1].size() << " (appended elements are zeros)!" <<std::endl;
                    N = results_mean[ilaunch - 1].size();
                }
                // Number of work groups
                numWorkGroups = results_mean[ilaunch - 1].size() / workGroupSize;

                // Result vector holding the (ilaunch+1)th kernel launch partial sums
                results_mean[ilaunch].resize(numWorkGroups);

                // Create buffers
                cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * results_mean[ilaunch - 1].size(), results_mean[ilaunch - 1].data());
                cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

                // mean kernel takes 4 arguments
                kernel_mean.setArg(0, buf_in);                                 //__global float* data
                kernel_mean.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
                kernel_mean.setArg(2, buf_out);                                //__global float* result
                kernel_mean.setArg(3, N_original);                             //int N_original_dataset


                // Start kernel and read the buf_out
                queue.enqueueNDRangeKernel(kernel_mean, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
                queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_mean[ilaunch].data());
            }
        }

        std::cout << "LOG: number of elements in last results_mean vector  = " << results_mean[n_launch - 1].size() << std::endl;
        float mean_of_data = results_mean[n_launch - 1][0];
        std::cout << "###############################" << std::endl;
        std::cout << "Mean of the input data = " << mean_of_data << std::endl;
        std::cout << "###############################\n" << std::endl;

        for(int ilaunch = 0; ilaunch < n_launch; ++ilaunch)
        {
            int numWorkGroups;
            // First kernel launch always needed
            if (ilaunch == 0)
            {
                numWorkGroups = data.size() / workGroupSize;
                
                // Result vector holding the first kernel launch partial var sums
                results_var[ilaunch].resize(numWorkGroups);

                // Create buffers
                cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * data.size(), data.data());
                cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

                // var_reduction kernel takes 4 arguments
                kernel_var_reduction.setArg(0, buf_in);                                 //__global float* data
                kernel_var_reduction.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
                kernel_var_reduction.setArg(2, buf_out);                                //__global float* result
                kernel_var_reduction.setArg(3, mean_of_data);                           //float mean

                // Start kernel and read the buf_out
                queue.enqueueNDRangeKernel(kernel_var_reduction, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
                queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_var[ilaunch].data());
            }

            // If more kernel launches are needed but not the last kernel launch: simply reduction problem
            else if ((ilaunch > 0) && (ilaunch < (n_launch - 1)))
            {
                // Check if N % workGroupSize is 0, if not, append 0s to data
                N = results_var[ilaunch - 1].size();
                if (N % workGroupSize != 0)
                {
                    results_var[ilaunch - 1].resize(N + (workGroupSize - (N % workGroupSize)), 0.0);
                    std::cout << "LOG: var computation ilaunch" <<ilaunch << ": resize was needed, from N = " << N << " to N = " << results_var[ilaunch - 1].size() << " (appended elements are zeros)!" <<std::endl;
                    N = results_var[ilaunch - 1].size();
                }
                // Number of work groups
                numWorkGroups = results_var[ilaunch - 1].size() / workGroupSize;

                // Result vector holding the (ilaunch+1)th kernel launch partial sums
                results_var[ilaunch].resize(numWorkGroups);

                // Create buffers
                cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * results_var[ilaunch - 1].size(), results_var[ilaunch - 1].data());
                cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

                // reduction kernel takes 3 arguments
                kernel_reduction.setArg(0, buf_in);                                 //__global float* data
                kernel_reduction.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
                kernel_reduction.setArg(2, buf_out);                                //__global float* result


                // Start kernel and read the buf_out
                queue.enqueueNDRangeKernel(kernel_reduction, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
                queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_var[ilaunch].data());

            }

            // Last kernel launch: reduction then division by N-1
            else if (ilaunch == (n_launch - 1))
            {
                // Check if N % workGroupSize is 0, if not, append 0s to data
                N = results_var[ilaunch - 1].size();
                if (N % workGroupSize != 0)
                {
                    results_var[ilaunch - 1].resize(N + (workGroupSize - (N % workGroupSize)), 0.0);
                    std::cout << "LOG: var computation ilaunch" <<ilaunch << ": resize was needed, from N = " << N << " to N = " << results_var[ilaunch - 1].size() << " (appended elements are zeros)!" <<std::endl;
                    N = results_var[ilaunch - 1].size();
                }
                // Number of work groups
                numWorkGroups = results_var[ilaunch - 1].size() / workGroupSize;

                // Result vector holding the (ilaunch+1)th kernel launch partial sums
                results_var[ilaunch].resize(numWorkGroups);

                // Create buffers
                cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * results_var[ilaunch - 1].size(), results_var[ilaunch - 1].data());
                cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

                // var kernel takes 4 arguments
                kernel_mean.setArg(0, buf_in);                                 //__global float* data
                kernel_mean.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
                kernel_mean.setArg(2, buf_out);                                //__global float* result
                kernel_mean.setArg(3, N_original -1);                          //for sample mean we need N-1

                // Start kernel and read the buf_out
                queue.enqueueNDRangeKernel(kernel_mean, cl::NullRange, cl::NDRange(N), cl::NDRange(workGroupSize));
                queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_var[ilaunch].data());
            }
        }
        std::cout << "LOG: number of elements in last results_var vector  = " << results_var[n_launch - 1].size() << std::endl;
        float var_of_data = results_var[n_launch - 1][0];
        std::cout << "###############################" << std::endl;
        std::cout << "Var of the input data = " << var_of_data << std::endl;
        std::cout << "###############################\n" << std::endl;

        


       
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
