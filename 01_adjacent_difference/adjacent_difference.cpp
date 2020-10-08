#include <OpenCL/opencl.hpp> // My written adjacent_difference kernel 
#include <vector>            // std::vector
#include <exception>         // std::runtime_error, std::exception
#include <iostream>          // std::cout
#include <fstream>           // std::ifstream
#include <random>            // std::default_random_engine, std::uniform_real_distribution
#include <cstdlib>           // EXIT_FAILURE
#include <chrono>            // std::chrono::high_resolution_clock::now()
#include <numeric>           // std::adjacent_difference

int main()
{
    std::cout << "main() started" << std::endl;
    try
    {
        // Get Queue, Device, Context, Platform
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Platform platform{device.getInfo<CL_DEVICE_PLATFORM>()};

        // Load adjacent_different.cl kernel source file
        std::ifstream source_file{ "../adjacent_difference.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "adjacent_difference.cl" };

        // Create cl::Program from kernel and build it for the device
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} } };
        program.build({ device });

        // My adjacent_difference function takes 2 args, that's why we put 2 cl::Buffers here
        auto adjacent_difference = cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "adjacent_difference");

        // Initialize computation
        constexpr cl::size_type N = 10000000; 
        std::vector<cl_float> vec_in(N), vec_out(N), vec_CPU_test(N);

        // Fill vec_in with random values between 0 and 100 (just to use this lovely random number generator)
        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_float>{ 0.0, 100.0 }]() mutable { return distribution(engine); };

        std::generate_n(std::begin(vec_in), N, prng);

        // Create buffers: we only want to read the vec_in, and we want to write the vec_out
        cl::Buffer buf_in{ std::begin(vec_in), std::end(vec_in), true };     
        cl::Buffer buf_out{ std::begin(vec_out), std::end(vec_out), false }; 

        // Dispatch of data before launch, a.k.a. copy the vec_in and vec_out into the cl::Buffers buf_in and buf_out
        cl::copy(queue, std::begin(vec_in), std::end(vec_in), buf_in);
        cl::copy(queue, std::begin(vec_out), std::end(vec_out), buf_out);

        // Launch adjacent_different calculator kernel and measure the computation time
        auto t0 = std::chrono::high_resolution_clock::now(); // Record start time

        adjacent_difference(cl::EnqueueArgs{ queue, cl::NDRange{ N } }, buf_in, buf_out);

        // Wait for the started kernel to finish
        cl::finish();

        auto t1 = std::chrono::high_resolution_clock::now(); // Record end time

        //Elapsed time while computing on GPU
        auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        // Fetch of results, a.k.a. copy the results from buf_out to vec_out
        cl::copy(queue, buf_out, std::begin(vec_out), std::end(vec_out));

        std::cout << "Elapsed computation time on GPU for N = "<< N << " long float vector: " << dt << " ms." << std::endl;

        // CPU computation time comparison 
        auto t0_CPU = std::chrono::high_resolution_clock::now();
        std::adjacent_difference(vec_in.begin(), vec_in.end(), vec_CPU_test.begin());
        auto t1_CPU= std::chrono::high_resolution_clock::now();
        auto dt_CPU = std::chrono::duration_cast<std::chrono::microseconds>(t1_CPU - t0_CPU).count();
        std::cout << "Elapsed computation time on CPU for N = "<< N << " long float vector: " << dt_CPU << " ms." << std::endl;

        // Check if my adjacent_difference calculator provides the same result as std::adjacent_difference
        if (std::equal(vec_out.begin(), vec_out.end(), vec_CPU_test.begin()))
            std::cout << "My adjacent_difference kernel provided the same results as we can get with std::adjacent_difference()." << std::endl;
        else
            std::cout << "The results of my adjacent_difference kernel and the results of std::adjacent_difference are not the same." << std::endl;
        
        

    }
    catch (cl::BuildError& error) // If kernel failed to build
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        for (const auto& log : error.getBuildLog())
        {
            std::cerr <<
                "\tBuild log for device: " <<
                log.first.getInfo<CL_DEVICE_NAME>() <<
                std::endl << std::endl <<
                log.second <<
                std::endl << std::endl;
        }

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
