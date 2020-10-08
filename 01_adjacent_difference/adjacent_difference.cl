__kernel void adjacent_difference(__global float* vec_in, __global float* vec_out)
{
    int gid = get_global_id(0);
    
    // First element of the vec_out is the first element of the vec_in
    if (gid == 0)
        vec_out[0] = vec_in[0];

    // Then every element is the pair-wise difference of indices (gid, gid-1)
    else 
        vec_out[gid] = vec_in[gid] - vec_in[gid-1];

}