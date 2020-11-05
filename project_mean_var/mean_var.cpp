#include "mean_var.hpp"

///------------------------------ FUNCTION DECLARATIONS ------------------------------///
int number_of_kernel_launches(int N_extended, int workGroupSize);
float compute_mean_gpu(cl::CommandQueue queue, cl::Context context, cl::Kernel kernel_reduction, cl::Kernel kernel_mean,
                       std::vector<float> data, std::vector< std::vector<float> > &results_mean,
                       int N_original, int N_extended, int n_launch, int workGroupSize);
float compute_var_gpu(cl::CommandQueue queue, cl::Context context,
                      cl::Kernel kernel_var_reduction, cl::Kernel kernel_reduction, cl::Kernel kernel_mean,
                      std::vector<float> data, std::vector< std::vector<float> > &results_var, 
                      int N_original, int N_extended, int n_launch, int workGroupSize, float mean_GPU);
void print_results(float mean, float var, bool gpu_results);
float compute_mean_cpu(std::vector<float> data_original, int N_original);
float compute_var_cpu(std::vector<float> data_original, int N_original, float mean_CPU);
void compare_cpu_gpu_results(float mean_CPU, float mean_GPU, float var_CPU, float var_GPU, float tolerance);

///------------------------------ MAIN ------------------------------///
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
        std::ifstream source_file_reduction{ "../reduction.cl" }; // kernel performing reduction
        std::ifstream source_file_mean{ "../mean.cl" }; // kernel performing reduction then dividing with given N
        std::ifstream source_file_var_reduction{ "../var_reduction.cl" }; // kernel performing reduction of (x_i - mean)**2
        if (!source_file_reduction.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "reduction.cl" };
        if (!source_file_mean.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "mean.cl" };
        if (!source_file_var_reduction.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "var_reduction.cl" };
    
        // Create cl::Program from kernels and build them for the device
        cl::Program program_reduction{ std::string{ std::istreambuf_iterator<char>{ source_file_reduction },
                                        std::istreambuf_iterator<char>{} } };
        cl::Program program_mean{ std::string{ std::istreambuf_iterator<char>{ source_file_mean },
                                        std::istreambuf_iterator<char>{} } };
        cl::Program program_var_reduction{ std::string{ std::istreambuf_iterator<char>{ source_file_var_reduction },
                                        std::istreambuf_iterator<char>{} } };
        program_reduction.build({ device });
        program_mean.build({ device });
        program_var_reduction.build({ device});

        // Create input data vector and fill with random numbers
        size_t N_original = 16777216;   // N should be between 2 - 2^24 (=16777216) for the project
        size_t N_extended = N_original; // After data extension, its value wont change (functions locally will modify, but won't affect global main scope)
        
        std::vector<float> data_original(N_original); // Original input data (needed in original shape for CPU calculations)
        std::vector<float> data_extended(N_extended); // Will be appended with 0s if size != workgroupsize * even number

        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_float>{ 0.0, 100.0 }]() mutable { return distribution(engine); };

        std::generate_n(std::begin(data_original), N_original, prng);
        data_extended = data_original;

        // Create kernels from programs
        cl::Kernel kernel_reduction(program_reduction, "reduction");
        cl::Kernel kernel_mean(program_mean, "mean");
        cl::Kernel kernel_var_reduction(program_var_reduction, "var_reduction");

        // Access work group size 
        int workGroupSize = kernel_reduction.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        
        // Check if N_original % workGroupSize is 0, if not, append 0s to data
        if (N_original % workGroupSize != 0)
        {
            N_extended = N_original + (workGroupSize - (N_original % workGroupSize));
            data_extended.resize(N_extended, 0.0);
            std::cout << "\nLOG: starting data, resize was needed, from N = " << N_original << " to N = " << N_extended << " (appended elements are zeros)!" <<std::endl;
        }
        
        // Determine how many kernel launches will be needed (based on N)
        int n_launch = number_of_kernel_launches(N_extended, workGroupSize);
        
        // Vectors for storing the results for each kernel launched in order to compute mean and var
        std::vector< std::vector<float> > results_mean(n_launch);
        std::vector< std::vector<float> > results_var(n_launch);

        // Compute mean using GPU
        float mean_GPU = compute_mean_gpu(queue, context, kernel_reduction, kernel_mean, data_extended, results_mean,
                                          N_original, N_extended, n_launch, workGroupSize);

        // Compute var using GPU
        float var_GPU = compute_var_gpu(queue, context, kernel_var_reduction, kernel_reduction, kernel_mean,
                                        data_extended, results_var, N_original, N_extended, n_launch, workGroupSize, mean_GPU);

        // Print out GPU results
        print_results(mean_GPU, var_GPU, true);
        
        // Perform mean and var CPU reference calculations
        float mean_CPU = compute_mean_cpu(data_original, N_original);
        float var_CPU = compute_var_cpu(data_original, N_original, mean_CPU);

        // Print out CPU results
        print_results(mean_CPU, var_CPU, false);

        // Check if mean and var computed by GPU and CPU are the same within small tolerance
        float tolerance = 1e-4;
        compare_cpu_gpu_results(mean_CPU, mean_GPU, var_CPU, var_GPU, tolerance);

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

    return 0;
}/// end of main

///------------------------------ FUNCTION DEFINITIONS ------------------------------///
int number_of_kernel_launches(int N_extended, int workGroupSize)
{
    int n_launch = 1;
    bool enough_launches = false;
    int N_temp = N_extended;
    
    while(not enough_launches)
    {
        if(N_temp / workGroupSize != 0)
        {
            N_temp /= workGroupSize;
            n_launch += 1;
        }
        else
            enough_launches = true;
    }
    std::cout << "LOG: number of required kernel launches = "<< n_launch <<" (for N = " << N_extended << ", work group size = " << workGroupSize << ")"<< std::endl;
    return n_launch;
}

float compute_mean_gpu(cl::CommandQueue queue, cl::Context context, cl::Kernel kernel_reduction, cl::Kernel kernel_mean,
                       std::vector<float> data, std::vector< std::vector<float> > &results_mean, 
                       int N_original, int N_extended, int n_launch, int workGroupSize)
{
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

            // Reduction kernel takes 3 arguments
            kernel_reduction.setArg(0, buf_in);                                 //__global float* data
            kernel_reduction.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
            kernel_reduction.setArg(2, buf_out);                                //__global float* result

            // Start kernel and read the buf_out
            queue.enqueueNDRangeKernel(kernel_reduction, cl::NullRange, cl::NDRange(N_extended), cl::NDRange(workGroupSize));
            queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_mean[ilaunch].data());
        }

        // If more kernel launches are needed but not the last kernel launch
        else if ((ilaunch > 0) && (ilaunch < (n_launch - 1)))
        {
            // Check if N % workGroupSize is 0, if not, append 0s to data
            N_extended = results_mean[ilaunch - 1].size();
            if (N_extended % workGroupSize != 0)
            {
                results_mean[ilaunch - 1].resize(N_extended + (workGroupSize - (N_extended % workGroupSize)), 0.0);
                std::cout << "LOG: mean computation ilaunch" <<ilaunch << ": resize was needed, from N = " << N_extended << " to N = " << results_mean[ilaunch - 1].size() << " (appended elements are zeros)!" <<std::endl;
                N_extended = results_mean[ilaunch - 1].size();
            }
            // Number of work groups
            numWorkGroups = results_mean[ilaunch - 1].size() / workGroupSize;

            // Result vector holding the (ilaunch+1)th kernel launch partial sums
            results_mean[ilaunch].resize(numWorkGroups);

            // Create buffers
            cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * results_mean[ilaunch - 1].size(), results_mean[ilaunch - 1].data());
            cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

            // Reduction kernel takes 3 arguments
            kernel_reduction.setArg(0, buf_in);                                 //__global float* data
            kernel_reduction.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
            kernel_reduction.setArg(2, buf_out);                                //__global float* result


            // Start kernel and read the buf_out
            queue.enqueueNDRangeKernel(kernel_reduction, cl::NullRange, cl::NDRange(N_extended), cl::NDRange(workGroupSize));
            queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_mean[ilaunch].data());

        }

        // Last kernel launch
        else if (ilaunch == (n_launch - 1))
        {
            // Check if N % workGroupSize is 0, if not, append 0s to data
            N_extended = results_mean[ilaunch - 1].size();
            if (N_extended % workGroupSize != 0)
            {
                results_mean[ilaunch - 1].resize(N_extended + (workGroupSize - (N_extended % workGroupSize)), 0.0);
                std::cout << "LOG: mean computation ilaunch" <<ilaunch << ": resize was needed, from N = " << N_extended << " to N = " << results_mean[ilaunch - 1].size() << " (appended elements are zeros)!" <<std::endl;
                N_extended = results_mean[ilaunch - 1].size();
            }
            // Number of work groups
            numWorkGroups = results_mean[ilaunch - 1].size() / workGroupSize;

            // Result vector holding the (ilaunch+1)th kernel launch partial sums
            results_mean[ilaunch].resize(numWorkGroups);

            // Create buffers
            cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * results_mean[ilaunch - 1].size(), results_mean[ilaunch - 1].data());
            cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

            // Mean kernel takes 4 arguments
            kernel_mean.setArg(0, buf_in);                                 //__global float* data
            kernel_mean.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
            kernel_mean.setArg(2, buf_out);                                //__global float* result
            kernel_mean.setArg(3, N_original);                             //int N

            // Start kernel and read the buf_out
            queue.enqueueNDRangeKernel(kernel_mean, cl::NullRange, cl::NDRange(N_extended), cl::NDRange(workGroupSize));
            queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_mean[ilaunch].data());
        }
    }
    float mean_GPU = results_mean[n_launch - 1][0];
    return mean_GPU;
}

float compute_var_gpu(cl::CommandQueue queue, cl::Context context,
                      cl::Kernel kernel_var_reduction, cl::Kernel kernel_reduction, cl::Kernel kernel_mean,
                      std::vector<float> data, std::vector< std::vector<float> > &results_var, 
                      int N_original, int N_extended, int n_launch, int workGroupSize, float mean_GPU)
{
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
            
            // Var reduction kernel takes 4 arguments
            kernel_var_reduction.setArg(0, buf_in);                                 //__global float* data
            kernel_var_reduction.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
            kernel_var_reduction.setArg(2, buf_out);                                //__global float* result
            kernel_var_reduction.setArg(3, mean_GPU);                               //float mean
            kernel_var_reduction.setArg(4, N_original);                             //int N_original

            // Start kernel and read the buf_out
            queue.enqueueNDRangeKernel(kernel_var_reduction, cl::NullRange, cl::NDRange(N_extended), cl::NDRange(workGroupSize));
            queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_var[ilaunch].data());
        }

        // If more kernel launches are needed but not the last kernel launch: simply reduction problem
        else if ((ilaunch > 0) && (ilaunch < (n_launch - 1)))
        {
            // Check if N % workGroupSize is 0, if not, append 0s to data
            N_extended = results_var[ilaunch - 1].size();
            if (N_extended % workGroupSize != 0)
            {
                results_var[ilaunch - 1].resize(N_extended + (workGroupSize - (N_extended % workGroupSize)), 0.0);
                std::cout << "LOG: var computation ilaunch" <<ilaunch << ": resize was needed, from N = " << N_extended << " to N = " << results_var[ilaunch - 1].size() << " (appended elements are zeros)!" <<std::endl;
                N_extended = results_var[ilaunch - 1].size();
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
            queue.enqueueNDRangeKernel(kernel_reduction, cl::NullRange, cl::NDRange(N_extended), cl::NDRange(workGroupSize));
            queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_var[ilaunch].data());

        }

        // Last kernel launch: reduction then division by N-1
        else if (ilaunch == (n_launch - 1))
        {
            // Check if N % workGroupSize is 0, if not, append 0s to data
            N_extended = results_var[ilaunch - 1].size();
            if (N_extended % workGroupSize != 0)
            {
                results_var[ilaunch - 1].resize(N_extended + (workGroupSize - (N_extended % workGroupSize)), 0.0);
                std::cout << "LOG: var computation ilaunch" <<ilaunch << ": resize was needed, from N = " << N_extended << " to N = " << results_var[ilaunch - 1].size() << " (appended elements are zeros)!" <<std::endl;
                N_extended = results_var[ilaunch - 1].size();
            }
            // Number of work groups
            numWorkGroups = results_var[ilaunch - 1].size() / workGroupSize;

            // Result vector holding the (ilaunch+1)th kernel launch partial sums
            results_var[ilaunch].resize(numWorkGroups);

            // Create buffers
            cl::Buffer buf_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * results_var[ilaunch - 1].size(), results_var[ilaunch - 1].data());
            cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * numWorkGroups);

            // Mean kernel takes 4 arguments
            kernel_mean.setArg(0, buf_in);                                 //__global float* data
            kernel_mean.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
            kernel_mean.setArg(2, buf_out);                                //__global float* result
            kernel_mean.setArg(3, N_original -1);                          //for sample mean we need N-1

            // Start kernel and read the buf_out
            queue.enqueueNDRangeKernel(kernel_mean, cl::NullRange, cl::NDRange(N_extended), cl::NDRange(workGroupSize));
            queue.enqueueReadBuffer(buf_out, true, 0, sizeof(float) * numWorkGroups, results_var[ilaunch].data());
        }
    }
    float var_GPU = results_var[n_launch - 1][0];
    return var_GPU;
}

void print_results(float mean, float var, bool gpu_results)
{
    if (gpu_results)
    {
        std::cout << "\n###############################" << std::endl;
        std::cout << "mean_GPU = " << mean << std::endl;
        std::cout << "var_GPU = " << var << std::endl;
        std::cout << "###############################\n" << std::endl;
    }
    else
    {
        std::cout << "\n###############################" << std::endl;
        std::cout << "mean_CPU = " << mean << std::endl;
        std::cout << "var_CPU = " << var << std::endl;
        std::cout << "###############################\n" << std::endl;
    }
}

float compute_mean_cpu(std::vector<float> data_original, int N_original)
{
    float sum_CPU = std::accumulate(data_original.begin(), data_original.end(), 0.0);
    float mean_CPU = sum_CPU / N_original;
    return mean_CPU;
}

float compute_var_cpu(std::vector<float> data_original, int N_original, float mean_CPU)
{
    std::vector<float> var_data(N_original);
    for(int i = 0; i < N_original; ++i)
        var_data[i] = (data_original[i] - mean_CPU) * (data_original[i] - mean_CPU);
    
    float sum_var_CPU = std::accumulate(var_data.begin(), var_data.end(), 0.0);
    float var_CPU = sum_var_CPU / (N_original - 1);
    return var_CPU;
}

void compare_cpu_gpu_results(float mean_CPU, float mean_GPU, float var_CPU, float var_GPU, float tolerance)
{
    float relative_error_mean = std::abs((mean_CPU - mean_GPU) / mean_CPU);
    float relative_error_var = std::abs((var_CPU - var_GPU) / var_CPU);

    std::cout << "###############################" << std::endl;
    std::cout << "Relative error for mean is: " << relative_error_mean << std::endl;
    std::cout << "Relative error for var is: " << relative_error_var << std::endl;

    if( relative_error_mean < tolerance )
        std::cout << "Mean calculation OK!" << std::endl;
    else
        std::cout << "Mean calculation WRONG!" << std::endl;

    if( relative_error_var < tolerance )
        std::cout << "Var calculation OK!" << std::endl;
    else
        std::cout << "Var calculation WRONG!" << std::endl;
    std::cout << "###############################\n" << std::endl;
}