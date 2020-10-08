__kernel void adjacent_difference(__global float* vec_in, __global float* vec_out)
{
    int gid = get_global_id(0);
    int N = 10;
    if (gid < N-1)
        vec_out[gid] = vec_in[gid] - vec_in[gid] + 8;
    else if (gid == N - 1)
        vec_out[gid] = 42;
}