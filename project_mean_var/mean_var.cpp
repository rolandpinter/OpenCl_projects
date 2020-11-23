// OpenCL include
#include <OpenCL/opencl.hpp> 

// Standard C++ includes
#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>

// Function to determine how many kernel launches will be needed based on the size of input data
int number_of_kernel_launches(size_t N, size_t workGroupSize, bool logging);

// Function to determine the sizes of the 3 buffers used for the reduction problems
std::vector<size_t> determine_buffer_sizes(size_t N, size_t workGroupSize, bool logging);

// Function to determine global work sizes of each kernel calls
std::vector<size_t> determine_global_work_sizes(int n_launch, size_t N, size_t workGroupSize, bool logging);

// Function to determine the size of data to be reduced during each kernel call
std::vector<size_t> determine_data_sizes_to_reduce(int n_launch, size_t N, size_t workGroupSize, bool logging);

// Function to call mean or var gpu kernels and handle the computations
float compute_mean_or_var_via_gpu(std::vector<cl::Buffer> vec_of_bufs, int n_launch, size_t N, size_t workGroupSize,
                                  std::vector<size_t> data_sizes_to_reduce, std::vector<size_t> global_work_sizes,
                                  cl::Kernel kernel, cl::CommandQueue queue, bool meanTrue_varFalse, float gpu_mean);

// Function to print out the results
void print_results(float mean, float var, bool gpu_results);

// Function to compute mean on CPU for reference calculation
float compute_mean_cpu(std::vector<float> data_original, int N_original);

// Function to compute mean on CPU for reference calculation
float compute_var_cpu(std::vector<float> data_original, int N_original, float mean_CPU);

// Function to compare CPU and GPU results, check if they are within epsilon tolerated range
void compare_cpu_gpu_results(float mean_CPU, float mean_GPU, float var_CPU, float var_GPU, float tolerance);

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
        std::ifstream source_file_mean{ "../mean_reduction.cl" }; // kernel performing sample mean calculation
        std::ifstream source_file_var{ "../var_reduction.cl" };   // kernel performing sample var calculation
        if (!source_file_mean.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "mean_reduction.cl" };
        if (!source_file_var.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "var_reduction.cl" };
    
        // Create cl::Program from kernels and build them for the device
        cl::Program program_mean{ std::string{ std::istreambuf_iterator<char>{ source_file_mean },
                                  std::istreambuf_iterator<char>{} } };
        cl::Program program_var{ std::string{ std::istreambuf_iterator<char>{ source_file_var },
                                 std::istreambuf_iterator<char>{} } };

        program_mean.build({ device });
        program_var.build({ device });

        // Create input data vector and fill with random numbers
        size_t N = 512*512*512 + 1;
        std::vector<float> data(N);

        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_float>{ 0.0, 100.0 }]() mutable { return distribution(engine); };

        std::generate_n(std::begin(data), N, prng);

        // Create kernels from programs
        cl::Kernel kernel_mean(program_mean, "mean_reduction");
        cl::Kernel kernel_var(program_var, "var_reduction");

        // Access work group size 
        size_t workGroupSize = kernel_mean.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        
        // Determine how many kernel launches will be needed (based on N)
        int n_launch = number_of_kernel_launches(N, workGroupSize, true);

        // Create a vector holding the 3 buffers required (buf1, buf2, buf3)
        std::vector<cl::Buffer> vec_of_bufs(3);

        // Determine buffer sizes
        std::vector<size_t>  buf_sizes = determine_buffer_sizes(N, workGroupSize, true);

        // Construct the buffers
        vec_of_bufs[0] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * buf_sizes[0], data.data());
        vec_of_bufs[1] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(float) * buf_sizes[1], nullptr);
        vec_of_bufs[2] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(float) * buf_sizes[2], nullptr);

        // Determine global work sizes for the started kernels
        std::vector<size_t> global_work_sizes = determine_global_work_sizes(n_launch, N, workGroupSize, true);

        // Determine data sizes to reduce during each kernel calls
        std::vector<size_t> data_sizes_to_reduce = determine_data_sizes_to_reduce(n_launch, N, workGroupSize, true);

        // Compute mean using GPU
        float gpu_mean = compute_mean_or_var_via_gpu(vec_of_bufs, n_launch, N, workGroupSize, data_sizes_to_reduce, global_work_sizes, kernel_mean, queue, true, 0.0);

        // Compute var using GPU
        float gpu_var = compute_mean_or_var_via_gpu(vec_of_bufs, n_launch, N, workGroupSize, data_sizes_to_reduce, global_work_sizes, kernel_var, queue, false, gpu_mean);

        // Print out GPU results
        print_results(gpu_mean, gpu_var, true);

        // Perform mean and var CPU reference calculations
        float cpu_mean = compute_mean_cpu(data, N);
        float cpu_var = compute_var_cpu(data, N, cpu_mean);
        
        // Print out CPU results
        print_results(cpu_mean, cpu_var, false);

        // Check if mean and var computed by GPU and CPU are the same within small tolerance
        float tolerance = 1e-6;
        compare_cpu_gpu_results(cpu_mean, gpu_mean, cpu_var, gpu_var, tolerance);

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

int number_of_kernel_launches(size_t N, size_t workGroupSize, bool logging)
{
    int n_launch = 1;
    bool enough_launches = false;
    int N_temp = N;
    
    while(!enough_launches)
    {
        if(N_temp / workGroupSize != 0)
        {
            N_temp /= workGroupSize;
            n_launch += 1;
        }
        else
            enough_launches = true;
    }
    if (logging)
        std::cout << "LOG: number of required kernel launches = "<< n_launch <<" (for N = " << N << ", work group size = " << workGroupSize << ")"<< std::endl;
    
    return n_launch;
}

std::vector<size_t> determine_buffer_sizes(size_t N, size_t workGroupSize, bool logging)
{
    // Determine sizes of buffers
    std::vector<size_t> vec_of_buf_sizes(3);

    // n1 = size of buf1 = size of input data
    // n2 = size of buf2 = how many work groups will handle the reduction of the input data
    // n3 = size of buf3 = how many work groups will handle the reduction of the result stored in buf2
    size_t n1 = N;

    size_t n2 = 0;
    if (N % workGroupSize == 0)
        n2 = N / workGroupSize;
    else
        n2 = N / workGroupSize + 1;
    
    size_t n3 = 0;
    if (n2 % workGroupSize == 0)
        n3 = n2 / workGroupSize;
    else
        n3 = n2 / workGroupSize + 1;

    vec_of_buf_sizes[0] = n1;
    vec_of_buf_sizes[1] = n2;
    vec_of_buf_sizes[2] = n3;
    
    if (logging)
    {
        std::cout << "LOG: Buffer sizes" << std::endl;
        std::cout << "\t n1 = " << n1 << std::endl;
        std::cout << "\t n2 = " << n2 << std::endl;
        std::cout << "\t n3 = " << n3 << std::endl;
    }
    
    return vec_of_buf_sizes;
}

std::vector<size_t> determine_global_work_sizes(int n_launch, size_t N, size_t workGroupSize, bool logging)
{
    // Determine enqueueNDRangeKernel's global work sizes
    std::vector<size_t> global_work_sizes(n_launch);

    // First element is the data size extended to the next (even number * work group size)
    if (N % workGroupSize == 0) 
        global_work_sizes[0] = N;
    else 
        global_work_sizes[0] = N + (workGroupSize - (N % workGroupSize));
    
    // Later elements are the previous element divided by work group size extended to the next (even number * work group size)
    for (int iLaunch = 1; iLaunch < n_launch; ++iLaunch)
    {
        if ((global_work_sizes[iLaunch - 1] / workGroupSize) % workGroupSize == 0) 
            global_work_sizes[iLaunch] = global_work_sizes[iLaunch - 1] / workGroupSize;
        else 
            global_work_sizes[iLaunch] = (global_work_sizes[iLaunch - 1] / workGroupSize) + (workGroupSize - ((global_work_sizes[iLaunch - 1] / workGroupSize) % workGroupSize));
    }

    if (logging)
    {
        std::cout << "LOG: Computed Global sizes" << std::endl;
        for(int iLaunch = 0; iLaunch < global_work_sizes.size(); ++iLaunch)
            std::cout << "\t iLaunch " << iLaunch << ": " << global_work_sizes[iLaunch] << std::endl;
    }

    return global_work_sizes;
}

std::vector<size_t> determine_data_sizes_to_reduce(int n_launch, size_t N, size_t workGroupSize, bool logging)
{
    // Determine data sizes to reduce during each kernel calls
    std::vector<size_t> data_sizes_to_reduce(n_launch);
    
    // First element is the size of the input data
    data_sizes_to_reduce[0] = N;

    for (int iLaunch = 1; iLaunch < n_launch; ++iLaunch)
    {
        if (data_sizes_to_reduce[iLaunch - 1] % workGroupSize == 0)
            data_sizes_to_reduce[iLaunch] = data_sizes_to_reduce[iLaunch - 1] / workGroupSize;
        else
            data_sizes_to_reduce[iLaunch] = data_sizes_to_reduce[iLaunch - 1] / workGroupSize + 1;
    }

    if (logging)
    {
        std::cout << "LOG: Number of data to reduce" << std::endl;
        for(int iLaunch = 0; iLaunch < n_launch; ++iLaunch)
            std::cout << "\t iLaunch " << iLaunch << ": " << data_sizes_to_reduce[iLaunch] << std::endl;
    }

    return data_sizes_to_reduce;
}

float compute_mean_or_var_via_gpu(std::vector<cl::Buffer> vec_of_bufs, int n_launch, size_t N, size_t workGroupSize,
                                  std::vector<size_t> data_sizes_to_reduce, std::vector<size_t> global_work_sizes,
                                  cl::Kernel kernel, cl::CommandQueue queue, bool meanTrue_varFalse, float gpu_mean)
{
    // Compute mean or var using GPU
    float mean_or_var_result = 0.0;
    
    // First kernel launch
    kernel.setArg(0, vec_of_bufs[0]);                         //__global float* data
    kernel.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData
    kernel.setArg(2, vec_of_bufs[1]);                         //__global float* result
    kernel.setArg(3, 0);                                      // int iLaunch
    kernel.setArg(4, n_launch - 1);                           // int lastLaunchIndex
    kernel.setArg(5, N);                                      // int N
    kernel.setArg(6, data_sizes_to_reduce[0]);                // int numOfValues
    
    // Next arg only needed for var calculating kernel
    if (meanTrue_varFalse == false)
        kernel.setArg(7, gpu_mean);                           // float mean
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_sizes[0]), cl::NDRange(workGroupSize));
    cl::finish();

    // Other kernel launches
    for(int iLaunch = 1; iLaunch < n_launch; ++iLaunch)
    { 
        if (iLaunch % 2 == 0)
            kernel.setArg(0, vec_of_bufs[2]);                     //__global float* data
        else
            kernel.setArg(0, vec_of_bufs[1]);                     //__global float* data
        
        kernel.setArg(1, sizeof(float) * workGroupSize, nullptr); //__local float* localData

        if (iLaunch % 2 == 0)
            kernel.setArg(2, vec_of_bufs[1]);                     //__global float* result
        else
            kernel.setArg(2, vec_of_bufs[2]);                     //__global float* result
        kernel.setArg(3, iLaunch);                                // int iLaunch
        kernel.setArg(4, n_launch - 1);                           // int lastLaunchIndex
        kernel.setArg(5, N);                                      // int N
        kernel.setArg(6, data_sizes_to_reduce[iLaunch]);          // int numOfValues

        // Next arg only needed for var calculating kernel
        if (meanTrue_varFalse == false)
            kernel.setArg(7, gpu_mean);                           // float mean
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_sizes[iLaunch]), cl::NDRange(workGroupSize));
        cl::finish();
    }
    // Read out the sample mean computed by GPU
    if (n_launch == 1 || n_launch % 2 == 1)
        queue.enqueueReadBuffer(vec_of_bufs[1], true, 0, sizeof(float) * 1, &mean_or_var_result);
    else if(n_launch > 1 || n_launch % 2 == 0)
        queue.enqueueReadBuffer(vec_of_bufs[2], true, 0, sizeof(float) * 1, &mean_or_var_result);

    return mean_or_var_result;
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