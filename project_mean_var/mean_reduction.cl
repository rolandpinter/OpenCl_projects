__kernel void mean_reduction(__global float* data, __local float* localData, __global float* result, int iLaunch, int lastLaunchIndex, size_t N, size_t numOfValues)
{
    int gid = get_global_id(0);        // id of the work item amongs every work items 
    int lid = get_local_id(0);         // id of the work item in the workgroup
    int localSize = get_local_size(0); // size of the workgroup

    // transfer from global to local memory
    // zero pad first
    localData[lid] = 0;
    // then if we have a valid value, copy it
    if (gid < numOfValues)
        localData[lid] = data[gid];

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

     if(lid == 0 && iLaunch != lastLaunchIndex)
     {
         // id of the workgroup = get_group_id()
        result[get_group_id(0)] = localData[0];
     }

     else if(lid == 0 && iLaunch == lastLaunchIndex)
     {
         result[get_group_id(0)] = localData[0] / N;
     }
}