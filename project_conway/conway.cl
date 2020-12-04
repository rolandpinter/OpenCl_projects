__kernel void conway(read_only image2d_t previous, write_only image2d_t next, unsigned int n, sampler_t grid_sampler)
{

    // x and y index of the cell we are about to work on
    int x = get_global_id(0);
    int y = get_global_id(1);

    /// Compute the number of living neighbors
    uint4 neighbors[9];

    neighbors[0] = read_imageui(previous, grid_sampler, (int2)(x, y));
    neighbors[1] = read_imageui(previous, grid_sampler, (int2)(x, y + 1));
    neighbors[2] = read_imageui(previous, grid_sampler, (int2)(x, y - 1));
    neighbors[3] = read_imageui(previous, grid_sampler, (int2)(x + 1, y));
    neighbors[4] = read_imageui(previous, grid_sampler, (int2)(x - 1, y));
    neighbors[5] = read_imageui(previous, grid_sampler, (int2)(x - 1, y + 1));
    neighbors[6] = read_imageui(previous, grid_sampler, (int2)(x - 1, y - 1));
    neighbors[7] = read_imageui(previous, grid_sampler, (int2)(x + 1, y + 1));
    neighbors[8] = read_imageui(previous, grid_sampler, (int2)(x + 1, y - 1));

    unsigned int n_living_neighbors = neighbors[0][0] + neighbors[1][0] + neighbors[2][0] +
                                      neighbors[3][0] + neighbors[4][0] + neighbors[5][0] +
                                      neighbors[6][0] + neighbors[7][0] + neighbors[8][0];
    /// Evaluate next state of cell
    // (1) No changes are made to the cell
    if (n_living_neighbors == n)
        write_imageui(next, int2(x,y), neighbors[0][0]);
        //write_imageui(next, int2(x,y), neighbors[0]);
    
    // (2) The cell will become alive
    else if(n_living_neighbors == (n + 1))
        write_imageui(next, int2(x,y), 1);
        //write_imageui(next, int2(x,y), {1, 0.0, 0.0, 1.0});

    // (3) The cell will become dead
    else
        write_imageui(next, int2(x,y), 0);
        //write_imageui(next, int2(x,y), {0, 0.0, 0.0, 1.0});

}