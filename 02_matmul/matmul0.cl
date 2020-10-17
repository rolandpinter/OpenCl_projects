__kernel void matmul0(__global double* A, 
                      __global double* B, 
                      __global double* C, 
                      int size)
{
  
   int thx = get_global_id(0); 
   int thy = get_global_id(1);

   double acc = 0.0;
   for (int i = 0; i < size; ++i)
   {
      acc += A[thy * size + i] * B[i * size + thx];
   }
 
   C[thy * size + thx] = acc;
}