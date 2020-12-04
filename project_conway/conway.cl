__kernel void conway(read_only image2d_t previous, write_only image2d_t next, sampler_t grid_sampler)
{
    // x and y index of the cell we are about to work on
    int x = get_global_id(0);
    int y = get_global_id(1);

    /// Compute the number of living neighbors
    int4 neighbors[8];
    neighbors[0] = read_imagei(previous, grid_sampler, (int2)(x, y + 1));
    neighbors[1] = read_imagei(previous, grid_sampler, (int2)(x, y - 1));
    neighbors[2] = read_imagei(previous, grid_sampler, (int2)(x + 1, y));
    neighbors[3] = read_imagei(previous, grid_sampler, (int2)(x - 1, y));
    neighbors[4] = read_imagei(previous, grid_sampler, (int2)(x - 1, y + 1));
    neighbors[5] = read_imagei(previous, grid_sampler, (int2)(x - 1, y - 1));
    neighbors[6] = read_imagei(previous, grid_sampler, (int2)(x + 1, y + 1));
    neighbors[7] = read_imagei(previous, grid_sampler, (int2)(x + 1, y - 1));

    int n_living_neighbors = neighbors[0][0] + neighbors[1][0] + neighbors[2][0] +
                             neighbors[3][0] + neighbors[4][0] + neighbors[5][0] +
                             neighbors[6][0] + neighbors[7][0];

    /// Get the current cell's status
    int4 current_cell = read_imagei(previous, grid_sampler, (int2)(x, y));
    int current_cell_state = current_cell[0];

    /// Evaluate next state of cell
    // (1) Any live cell with fewer than two live neighbours dies, as if by underpopulation
    if (current_cell_state == 1 && n_living_neighbors < 2)
        write_imagei(next, int2(x,y), (int4)(0, 0, 0, 0));

    // (2) Any live cell with two or three live neighbours lives on to the next generation
    else if (current_cell_state == 1 && (n_living_neighbors == 2 || n_living_neighbors == 3))
        write_imagei(next, int2(x,y), (int4)(1, 0, 0, 0));
    
    // (3) Any live cell with more than three live neighbours dies, as if by overpopulation
    else if (current_cell_state == 1 && n_living_neighbors > 3)
        write_imagei(next, int2(x,y), (int4)(0, 0, 0, 0));

    // (4) Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction
    else if (current_cell_state == 0 && n_living_neighbors == 3)
        write_imagei(next, int2(x,y), (int4)(1, 0, 0, 0));

    // (5) Any dead cell with less or more than three live neighbours dies
    else if (current_cell_state == 0 && n_living_neighbors != 3)
        write_imagei(next, int2(x,y), (int4)(0, 0, 0, 0));
}