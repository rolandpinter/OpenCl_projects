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

int main()
{
    try
    {
        /// GPU usual inits: queue, device, platform, context
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Platform platform{device.getInfo<CL_DEVICE_PLATFORM>()};

        /// Load kernel source file
        std::ifstream source_file{ "../conway.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "conway.cl" };
        
        /// Create cl::Program from kernel and build it for the device
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                             std::istreambuf_iterator<char>{} } };
        program.build({ device });

        // Create kernel from program
        cl::Kernel kernel(program, "conway");

        /// Init N and n parameters of the game of life
        size_t N;       // Grid size of the game of life (N X N grid)
        unsigned int n; // How many neighbors should we take into account (n = [0,8])

        std::cout << "Grid size, N = ";
        std::cin >> N;

        std::cout << "Number of neighbors to take into account (between 0 and 8), n = ";
        std::cin >> n;

        /// Create grid with random 0 and 1 values
        std::random_device rd;                       // Only used once to initialise (seed) engine
        std::mt19937 rng(rd());                      // Random-number engine used (Mersenne-Twister)
        std::uniform_int_distribution<int> uni(0,1); // Uniformly and randomly 0s and 1s 

        std::vector<std::vector<unsigned int> > state_of_game(N, std::vector<unsigned int>(N)); // N*N grid's representation: N*N vector
        for(int i=0;i<N;++i)
            for(int j=0;j<N;++j)
                state_of_game[i][j] = uni(rng);

        /// Print out the starting state of the game's grid
        std::cout << "\n \n STARTING STATE" << std::endl;
        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N; ++j)
                std::cout << state_of_game[i][j] << " ";
            std::cout<<"\n";
        }

        /// Parameters of the textures to be created
        size_t width = N;
        size_t height = N;
        
        /// Create a vector holding the 2 required textures
        std::vector<cl::Image2D> vec_of_textures(2);
        vec_of_textures[0] = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), width, height,
                                         0, state_of_game.data(), nullptr);

        vec_of_textures[1] = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                         cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), width, height,
                                         0, state_of_game.data(), nullptr);

        /// Create cl::Sampler
        cl::Sampler sampler = cl::Sampler(context, CL_FALSE, CL_ADDRESS_REPEAT, CL_FILTER_NEAREST);

        /// Next step: call kernel and check if .cl code does what is should be doing
        // First kernel launch

        kernel.setArg(0, vec_of_textures[0]);
        kernel.setArg(1, vec_of_textures[1]);
        kernel.setArg(2, n);
        kernel.setArg(3, sampler);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N,N), cl::NullRange);

        const std::array<cl::size_type, 3> origin = {0,0,0};
        const std::array<cl::size_type, 3> region = {N, N, 1};
        std::vector<std::vector<unsigned int> > output(N, std::vector<unsigned int>(N));

        queue.enqueueReadImage(vec_of_textures[1], false, origin, region, 0, 0, output.data(), 0, nullptr);
        cl::finish();

        std::cout << "\n \n SECOND STATE" << std::endl;
        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N; ++j)
                std::cout << output[i][j] << " ";
            std::cout<<"\n";
        }



        

        



























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