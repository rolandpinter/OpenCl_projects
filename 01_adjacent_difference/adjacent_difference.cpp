#include <OpenCL/opencl.hpp>

#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE

int main()
{
    try
    {
        // Get Queue, Device, Context, Platform
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Platform platform{device.getInfo<CL_DEVICE_PLATFORM>()};

        std::cout << "Default queue on platform: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        std::cout << "Default queue on device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Load program source
        std::ifstream source_file{ "../adjacent_difference.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "adjacent_difference.cl" };

        // Create program and kernel

        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} } };
        program.build({ device });

        auto adjacent_difference = cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "adjacent_difference");

        // Init computation
        constexpr cl::size_type N = 10;
        std::vector<cl_float> vec_in(N), vec_out(N);

        // Fill vec_in with random values between 0 and 100, just to use this lovel random number generator
        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_float>{ 0.0, 100.0 }]() mutable { return distribution(engine); };

        std::generate_n(std::begin(vec_in), N, prng);

        // Create buffers: we only want to read the vec_in, and we want to write the vec_out!
        cl::Buffer buf_in{ std::begin(vec_in), std::end(vec_in), true };     
        cl::Buffer buf_out{ std::begin(vec_out), std::end(vec_out), false }; 

        
        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(vec_in), std::end(vec_in), buf_in);
        cl::copy(queue, std::begin(vec_out), std::end(vec_out), buf_out);

        // Launch kernels
        adjacent_difference(cl::EnqueueArgs{ queue, cl::NDRange{ N } }, buf_in, buf_out);

        cl::finish();

        // (Blocking) fetch of results
        cl::copy(queue, buf_out, std::begin(vec_out), std::end(vec_out));

        for (int i=0;i<vec_out.size();++i)
            std::cout<<vec_out[i]<<std::endl;

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
