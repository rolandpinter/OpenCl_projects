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

void dump_state_of_game(char* file_base_name, unsigned int t, size_t N, std::vector<int> state_of_game);

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

        /// Create kernel from program
        cl::Kernel kernel(program, "conway");

        /// Init N parameter of the game: the game is played on an N * N big square grid
        size_t N = 64;

        ///Init T parameter of the game: how many iterations we do
        unsigned int T = 300;

        /// Fill grid with random cell states or with pre-defined one
        bool random_starting_state = false;

        /// Vector holding the state of the game
        std::vector<int> state_of_game(N * N);

        if (random_starting_state)
        {
            /// Create grid with random 0 and 1 values
            std::random_device rd;                        // Only used once to initialise (seed) engine
            std::mt19937 rng(rd());                       // Random-number engine used (Mersenne-Twister)
            std::uniform_int_distribution<int> uni(0, 1); // Uniformly and randomly 0s and 1s 

            for(int i = 0; i < (N * N); ++i)
                    state_of_game[i] = uni(rng);
        }

        else
        {
            for(int i = 0; i < N*N; ++ i)
            {
                // Glider gun
                if(i == (N + 25)) state_of_game[i] = 1;
                else if(i == (2*N + 23)) state_of_game[i] = 1;
                else if(i == (2*N + 25)) state_of_game[i] = 1;
                else if(i == (3*N + 13)) state_of_game[i] = 1;
                else if(i == (3*N + 14)) state_of_game[i] = 1;
                else if(i == (3*N + 21)) state_of_game[i] = 1;
                else if(i == (3*N + 22)) state_of_game[i] = 1;
                else if(i == (3*N + 35)) state_of_game[i] = 1;
                else if(i == (3*N + 36)) state_of_game[i] = 1;
                else if(i == (4*N + 12)) state_of_game[i] = 1;
                else if(i == (4*N + 16)) state_of_game[i] = 1;
                else if(i == (4*N + 21)) state_of_game[i] = 1;
                else if(i == (4*N + 22)) state_of_game[i] = 1;
                else if(i == (4*N + 35)) state_of_game[i] = 1;
                else if(i == (4*N + 36)) state_of_game[i] = 1;
                else if(i == (5*N + 1)) state_of_game[i] = 1;
                else if(i == (5*N + 2)) state_of_game[i] = 1;
                else if(i == (5*N + 11)) state_of_game[i] = 1;
                else if(i == (5*N + 17)) state_of_game[i] = 1;
                else if(i == (5*N + 21)) state_of_game[i] = 1;
                else if(i == (5*N + 22)) state_of_game[i] = 1;
                else if(i == (6*N + 1)) state_of_game[i] = 1;
                else if(i == (6*N + 2)) state_of_game[i] = 1;
                else if(i == (6*N + 11)) state_of_game[i] = 1;
                else if(i == (6*N + 15)) state_of_game[i] = 1;
                else if(i == (6*N + 17)) state_of_game[i] = 1;
                else if(i == (6*N + 18)) state_of_game[i] = 1;
                else if(i == (6*N + 23)) state_of_game[i] = 1;
                else if(i == (6*N + 25)) state_of_game[i] = 1;
                else if(i == (7*N + 11)) state_of_game[i] = 1;
                else if(i == (7*N + 17)) state_of_game[i] = 1;
                else if(i == (7*N + 25)) state_of_game[i] = 1;
                else if(i == (8*N + 12)) state_of_game[i] = 1;
                else if(i == (8*N + 16)) state_of_game[i] = 1;
                else if(i == (9*N + 13)) state_of_game[i] = 1;
                else if(i == (9*N + 14)) state_of_game[i] = 1;
                
                // Others are dead
                else state_of_game[i] = 0;
            }
        }
        
        /// Parameters of the textures to be created
        size_t width = N;
        size_t height = N;
        
        /// Create a vector holding the 2 required textures
        std::vector<cl::Image2D> vec_of_textures(2);
        vec_of_textures[0] = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         cl::ImageFormat(CL_R, CL_SIGNED_INT32), width, height,
                                         0, state_of_game.data(), nullptr);

        vec_of_textures[1] = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY |CL_MEM_COPY_HOST_PTR,
                                         cl::ImageFormat(CL_R, CL_SIGNED_INT32), width, height,
                                         0, state_of_game.data(), nullptr);

        /// Create cl::Sampler
        cl::Sampler sampler = cl::Sampler(context, CL_FALSE, CL_ADDRESS_REPEAT, CL_FILTER_NEAREST);

        /// Offset and size arrays for enqueueReadImage()
        const std::array<cl::size_type, 3> origin = {0,0,0};
        const std::array<cl::size_type, 3> region = {N, N, 1};

        /// File base name for dumping out the state of the game
        char file_base_name[] = "../csv_outputs/grid"; 

        /// Play the game T times
        for(unsigned int t = 0; t < T; ++t)
        {
            /// Print out the state of the game csv files
            dump_state_of_game(file_base_name, t, N, state_of_game);
            /// Set kernel arguments
            if (t % 2 == 0)
            {
                kernel.setArg(0, vec_of_textures[0]);
                kernel.setArg(1, vec_of_textures[1]);
            }
            else
            {
                kernel.setArg(0, vec_of_textures[1]);
                kernel.setArg(1, vec_of_textures[0]);
            }
            kernel.setArg(2, sampler);

            /// Launch kernel
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N,N), cl::NullRange);
            cl::finish();

            /// Read the state of the game
            if (t % 2 == 0)
                queue.enqueueReadImage(vec_of_textures[1], true, origin, region, 0, 0, state_of_game.data(), 0, nullptr);
            else
                queue.enqueueReadImage(vec_of_textures[0], true, origin, region, 0, 0, state_of_game.data(), 0, nullptr);
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

void dump_state_of_game(char* file_base_name, unsigned int t, size_t N, std::vector<int> state_of_game)
{
    std::stringstream outpath;
    outpath << file_base_name << t << ".csv";

    std::ofstream file(outpath.str().c_str());

    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            if(state_of_game[i * N + j] == 1) file << "1";
            else if(state_of_game[i * N + j] == 0) file << "0";
            if (j < N - 1) file << ",";
        }
        file << "\n";
   }
   file.close();
}