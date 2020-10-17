// OpenCL includes
#include <OpenCL/opencl.hpp> 

// Standard C++ includes
#include <sstream>
#include <fstream>
#include <algorithm>
#include <memory>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>
#include <chrono>


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

        // Load matmul kernel source files
        std::ifstream source_file_matmul0{ "../matmul0.cl" };
		std::ifstream source_file_matmul1{ "../matmul1.cl" };

		// Check if kernel source files were opened or not
        if (!source_file_matmul0.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "matmul0.cl" };
		if (!source_file_matmul1.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "matmul1.cl" };

        // Create cl::Program from kernels and build them for the device
        cl::Program program_matmul0{ std::string{ std::istreambuf_iterator<char>{ source_file_matmul0 },
                                          std::istreambuf_iterator<char>{} } };
        cl::Program program_matmul1{ std::string{ std::istreambuf_iterator<char>{ source_file_matmul1 },
                                          std::istreambuf_iterator<char>{} } };
        program_matmul0.build({ device });
        program_matmul1.build({ device });

/*TODO: matmul1 from here on is missing*/

        //  Create KernelFunctors for the kernels 
        auto matmul0 = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int>(program_matmul0, "matmul0");

        // Initialize computation
        constexpr int size = 1024; 
        std::vector<float> A(size*size), B(size*size);
        std::vector<float> matmul0_result_GPU(size*size), matmul0_result_CPU(size*size);

        // Fill A and B matrix with random values between -1 and 1 and result matrices with zeros
        std::random_device rnd_device;
	    std::mt19937 mersenne_engine(rnd_device());
	    std::uniform_real_distribution<double> dist(-1.0, 1.0);
	    auto gen = [&]() { return dist(mersenne_engine); };

        std::generate(A.begin(), A.end(), gen);
	    std::generate(B.begin(), B.end(), gen);
        std::fill(matmul0_result_GPU.begin(), matmul0_result_GPU.end(), 0.0f);
        std::fill(matmul0_result_CPU.begin(), matmul0_result_CPU.end(), 0.0f);

        // Create buffers: we only want to read the A and B matrices, and we want to write the C matrix
        cl::Buffer buf_A{ std::begin(A), std::end(A), true };
        cl::Buffer buf_B{ std::begin(B), std::end(B), true };    
        cl::Buffer buf_matmul0_result{ std::begin(matmul0_result_GPU), std::end(matmul0_result_GPU), false }; 

        // Dispatch of data before launch, a.k.a. copy the A, B, C into the cl::Buffers buf_A, buf_B, buf_C
        cl::copy(queue, std::begin(A), std::end(A), buf_A);
        cl::copy(queue, std::begin(B), std::end(B), buf_B);
        cl::copy(queue, std::begin(matmul0_result_GPU), std::end(matmul0_result_GPU), buf_matmul0_result);

        // Launch matmul0  kernel and measure the computation time
        auto tStart_matmul0_GPU = std::chrono::high_resolution_clock::now(); // Record start time

        matmul0(cl::EnqueueArgs{ queue, cl::NDRange{ size*size } }, buf_A, buf_B, buf_matmul0_result, size);

        // Wait for the started kernel to finish
        cl::finish();

        auto tEnd_matmul0_GPU = std::chrono::high_resolution_clock::now(); // Record end time

        //Elapsed time while computing on GPU
        auto dt_matmul0_GPU = std::chrono::duration_cast<std::chrono::microseconds>(tEnd_matmul0_GPU - tStart_matmul0_GPU).count();
        std::cout << "matmul0 GPU computation time : " << dt_matmul0_GPU << " ms." << std::endl;
        // Fetch of results, a.k.a. copy the results from buf_matmul0_result to matmul0_result_GPU
        cl::copy(queue, buf_matmul0_result,  std::begin(matmul0_result_GPU), std::end(matmul0_result_GPU));


        // Calculate matmul0 style with CPU as well
        auto tStart_matmul0_CPU = std::chrono::high_resolution_clock::now(); // Record start time
        for(int i = 0; i < size; ++i)
        {
            for(int j = 0; j < size; ++j)
            {
                float foo = 0.0;
                for(int k = 0; k < size; ++k)
                {
                    foo += A[i*size + k] * B[k*size + j];
                }
                matmul0_result_CPU[i*size + j] = foo;
            }
        }
        auto tEnd_matmul0_CPU = std::chrono::high_resolution_clock::now(); // Record end time
        auto dt_matmul0_CPU = std::chrono::duration_cast<std::chrono::microseconds>(tEnd_matmul0_CPU - tStart_matmul0_CPU).count();
        std::cout << "matmul0 CPU computation time : " << dt_matmul0_CPU << " ms." << std::endl;

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
