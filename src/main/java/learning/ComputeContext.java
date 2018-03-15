package learning;

import org.jocl.*;
import static org.jocl.CL.*;
import learning.tensor.*;

import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

import java.util.ArrayList;

public class ComputeContext {
   
   public static cl_platform_id platform;
   public static cl_device_id device;
   public static cl_context context;
   public static cl_command_queue queue;
   
   private static cl_kernel kernel_copy, kernel_add, kernel_radd, kernel_zeros, kernel_ones;
   
   private final static ArrayList kernels = new ArrayList();
   private final static ArrayList programs = new ArrayList();
   
   public static long pref_local_size = 0L;
   
   public static boolean OPEN_CL = false, PROFILING = false, RECURRENT = false;
   
   public static boolean init() {
      // The platform, device type and device number
      // that will be used
      final int platformIndex = 0;
      final long deviceType = CL_DEVICE_TYPE_ALL;
      final int deviceIndex = 0;
      
      // Enable exceptions and subsequently omit error checks in this sample
      CL.setExceptionsEnabled(true);
      
      // Obtain the number of platforms
      int numPlatformsArray[] = new int[1];
      clGetPlatformIDs(0, null, numPlatformsArray);
      int numPlatforms = numPlatformsArray[0];
      
      // Obtain a platform ID
      cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
      clGetPlatformIDs(platforms.length, platforms, null);
      platform = platforms[platformIndex];
      
      // Initialize the context properties
      cl_context_properties contextProperties = new cl_context_properties();
      contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
      
      // Obtain the number of devices for the platform
      int numDevicesArray[] = new int[1];
      clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
      int numDevices = numDevicesArray[0];
      
      // Obtain a device ID
      cl_device_id devices[] = new cl_device_id[numDevices];
      clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
      device = devices[deviceIndex];
      
      // Create a context for the selected device
      context = clCreateContext(contextProperties, 1, new cl_device_id[] {device}, null, null, null);
      
      // Create a command-queue for the selected device
      queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, null);
      
      kernel_copy = getKernel("./kernel/copy.c", "copy");
      kernel_add = getKernel("./kernel/add.c", "add");
      kernel_radd = getKernel("./kernel/pointadd.c", "pointadd");
      kernel_zeros = getKernel("./kernel/zeros.c", "zeros");
      kernel_ones = getKernel("./kernel/ones.c", "ones");
      
      return true;
   }
   
   public static long[] getDeviceInfo(int flag) {
      long values[] = new long[4];
      clGetDeviceInfo(device, flag, Sizeof.cl_long * 4, Pointer.to(values), null);
      return values;
   }
   
   public static void release() {
      for(int i = 0; i<kernels.size(); i++) {
         clReleaseKernel((cl_kernel) kernels.get(i));
         clReleaseProgram((cl_program) programs.get(i));
      }
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
      clReleaseDevice(device);
   }
   
   public static void copy(Tensor x, Tensor y) {
      //y = x;
      clSetKernelArg(kernel_copy, 0, Sizeof.cl_mem, Pointer.to(x.getMem()));
      clSetKernelArg(kernel_copy, 1, Sizeof.cl_mem, Pointer.to(y.getMem()));
      
      clEnqueueNDRangeKernel(queue, kernel_copy, 1, null, y.global_size, y.local_size, 0, null, null);
   }
   
   public static void add(Tensor x, Tensor y) {
      //y += x
      clSetKernelArg(kernel_add, 0, Sizeof.cl_mem, Pointer.to(x.getMem()));
      clSetKernelArg(kernel_add, 1, Sizeof.cl_mem, Pointer.to(y.getMem()));
      
      clEnqueueNDRangeKernel(queue, kernel_add, 1, null, y.global_size, y.local_size, 0, null, null);
   }
   
   public static void add(Tensor x1, Tensor x2, Tensor y) {
      //y = x1 + x2;
      clSetKernelArg(kernel_radd, 0, Sizeof.cl_mem, Pointer.to(x1.getMem()));
      clSetKernelArg(kernel_radd, 1, Sizeof.cl_mem, Pointer.to(x2.getMem()));
      clSetKernelArg(kernel_radd, 2, Sizeof.cl_mem, Pointer.to(y.getMem()));
      
      clEnqueueNDRangeKernel(queue, kernel_radd, 1, null, y.global_size, y.local_size, 0, null, null);
   }
   
   public static void zeros(Tensor y) {
      clSetKernelArg(kernel_zeros, 0, Sizeof.cl_mem, Pointer.to(y.getMem()));
      
      clEnqueueNDRangeKernel(queue, kernel_zeros, 1, null, y.global_size, y.local_size, 0, null, null);
   }
   
   public static void ones(Tensor y) {
      clSetKernelArg(kernel_ones, 0, Sizeof.cl_mem, Pointer.to(y.getMem()));
      
      clEnqueueNDRangeKernel(queue, kernel_ones, 1, null, y.global_size, y.local_size, 0, null, null);
   }
   
   public static cl_kernel getKernel(String path, String name) {
      String source;
      try {
         byte[] encoded = Files.readAllBytes(Paths.get(path));
         source = new String(encoded, Charset.defaultCharset());
      } catch(IOException ex) {
         System.err.println(ex);
         return null;
      }
      
      if(source != null) {
         cl_program program = clCreateProgramWithSource(context, 1, new String[] {source}, null, null);
         clBuildProgram(program, 0, null, null, null, null);
         cl_kernel kernel = clCreateKernel(program, name, null);
         
         if(pref_local_size == 0L) {
            long values[] = new long[1];
            clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, Sizeof.cl_long, Pointer.to(values), null);
            pref_local_size = values[0];
         }
         
         programs.add(program);
         kernels.add(kernel);
         return kernel;
      }
      return null;
   }
   
   public static void addKernel(cl_program p, cl_kernel k) {
      programs.add(p);
      kernels.add(k);
   }
}