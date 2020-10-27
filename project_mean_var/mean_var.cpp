#include <OpenCL/opencl.hpp> // OpenCl Header 
#include <vector>            // std::vector
#include <exception>         // std::runtime_error, std::exception
#include <iostream>          // std::cout
#include <fstream>           // std::ifstream
#include <random>            // std::default_random_engine, std::uniform_real_distribution
#include <cstdlib>           // EXIT_FAILURE
#include <chrono>            // std::chrono::high_resolution_clock::now()

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
        std::ifstream source_file{ "../mean_var.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "mean_var.cl" };

        // Create cl::Program from kernel and build it for the device
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} } };
        program.build({ device });




        
        

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
