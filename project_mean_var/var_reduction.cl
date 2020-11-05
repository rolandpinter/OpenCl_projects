__kernel void var_reduction(__global float* data, __local float* localData, __global float* result, float mean, int N_original) 
{
    int gid = get_global_id(0);        // id of the work item amongs every work items 
    int lid = get_local_id(0);         // id of the work item in the workgroup
    int localSize = get_local_size(0); // size of the workgroup

    // transfer from global to local memory
    // data was appended with 0s to match work group size, and these appended zeros make var calculation wrong
    // so first, transfer 0 values, and if a value was not an appended 0, a.k.a. gid < N_original, transfer (x_i - mean)**2 as needed
    if (gid < N_original)
        localData[lid] = ((data[gid] - mean) * (data[gid] - mean));
    else
        localData[lid] = 0;
    
    // make sure everything up to this point in the workgroup finished executing
    barrier(CLK_LOCAL_MEM_FENCE);

    // perform reduction in local memory
    for(unsigned int s = 1; s < localSize; s *= 2)
    {
        // sum every adjacent pairs
        if (lid % (2 * s) == 0)
        {
            localData[lid] += localData[lid + s];
        }

        // make sure everything is synchronized properly before we go into the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

     if(lid == 0)
     {
        // id of the workgroup = get_group_id()
        result[get_group_id(0)] =  localData[0];
     }
}