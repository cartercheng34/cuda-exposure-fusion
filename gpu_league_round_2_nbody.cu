//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 2: n-body simulation
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team = "TaiwanNo1";
	std::string author_1 = "Hsu_Cheng";
	std::string author_2 = "Name_2";
	std::string author_3 = "Name_3";	////optional
};

//////////////////////////////////////////////////////////////////////////
////Here is a sample function implemented on CPU for n-body simulation.

__host__ void N_Body_Simulation_CPU_Poorman(double* pos_x, double* pos_y, double* pos_z,		////position array
	double* vel_x, double* vel_y, double* vel_z,		////velocity array
	double* acl_x, double* acl_y, double* acl_z,		////acceleration array
	const double* mass,								////mass array
	const int n,									////number of particles
	const double dt,								////timestep
	const double epsilon_squared)					////epsilon to avoid 0-denominator
{
	////Step 1: set particle accelerations to be zero
	memset(acl_x, 0x00, sizeof(double) * n);
	memset(acl_y, 0x00, sizeof(double) * n);
	memset(acl_z, 0x00, sizeof(double) * n);

	////Step 2: traverse all particle pairs and accumulate gravitational forces for each particle from pairwise interactions
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			////skip calculating force for itself
			if (i == j) continue;

			////r_ij=x_j-x_i
			double rx = pos_x[j] - pos_x[i];
			double ry = pos_y[j] - pos_y[i];
			double rz = pos_z[j] - pos_z[i];

			////a_ij=m_j*r_ij/(r+epsilon)^3, 
			////noticing that we ignore the gravitational coefficient (assuming G=1)
			double dis_squared = rx * rx + ry * ry + rz * rz;
			double one_over_dis_cube = 1.0 / pow(sqrt(dis_squared + epsilon_squared), 3);
			double ax = mass[j] * rx * one_over_dis_cube;
			double ay = mass[j] * ry * one_over_dis_cube;
			double az = mass[j] * rz * one_over_dis_cube;

			////accumulate the force to the particle
			acl_x[i] += ax;
			acl_y[i] += ay;
			acl_z[i] += az;
		}
	}
	//debug
	/*double rx = pos_x[4095] - pos_x[2048];
	double ry = pos_y[4095] - pos_y[2048];
	double rz = pos_z[4095] - pos_z[2048];
	double dis_squared = rx * rx + ry * ry + rz * rz;
	double one_over_dis_cube = 1.0 / pow(sqrt(dis_squared + epsilon_squared), 3);

	printf("x acl: %.6f\n", pos_x[4095]);
	printf("y acl: %.6f\n", pos_y[4095]);
	printf("z acl: %.6f\n", pos_z[4095]);*/


	////Step 3: explicit time integration to update the velocity and position of each particle
	for (int i = 0; i < n; i++) {
		////v_{t+1}=v_{t}+a_{t}*dt
		vel_x[i] += acl_x[i] * dt;
		vel_y[i] += acl_y[i] * dt;
		vel_z[i] += acl_z[i] * dt;

		////x_{t+1}=x_{t}+v_{t}*dt
		pos_x[i] += vel_x[i] * dt;
		pos_y[i] += vel_y[i] * dt;
		pos_z[i] += vel_z[i] * dt;
	}
}


//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here





////Your implementations end here
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
////Test function for n-body simulator
ofstream out;

//////////////////////////////////////////////////////////////////////////
////Please do not change the values below
const double dt = 0.001;							////time step
const int time_step_num = 10;						////number of time steps
const double epsilon = 1e-2;						////epsilon added in the denominator to avoid 0-division when calculating the gravitational force
const double epsilon_squared = epsilon * epsilon;	////epsilon squared

////We use grid_size=4 to help you debug your code, change it to a bigger number (e.g., 16, 32, etc.) to test the performance of your GPU code
const unsigned int grid_size = 16;					////assuming particles are initialized on a background grid
const unsigned int particle_n = pow(grid_size, 3);	////assuming each grid cell has one particle at the beginning


const int threads_num = 256;
const int block_num = (particle_n / 256) < 1 ? 1 : particle_n / 256;



__global__ void compute_acl(double* pos_x, double* pos_y, double* pos_z,		////position array
	double* vel_x, double* vel_y, double* vel_z,		////velocity array
	double* acl_x, double* acl_y, double* acl_z,		////acceleration array
	const double* mass,								////mass array
	const int n,									////number of particles
	const double dt,								////timestep
	const double epsilon_squared)					////epsilon to avoid 0-denominator 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//const int threads_num = 64;

	__shared__ double localPosX[threads_num];
	__shared__ double localPosY[threads_num];
	__shared__ double localPosZ[threads_num];

	__shared__ double localmass[threads_num];

	// self position
	double3 body = { pos_x[tid],
				   pos_y[tid],
				   pos_z[tid] };

	double3 acl = { 0.0f, 0.0f, 0.0f };




	for (unsigned int tile = 0; tile < n; tile += threads_num) {


		localPosX[threadIdx.x] = pos_x[tile + threadIdx.x];
		localPosY[threadIdx.x] = pos_y[tile + threadIdx.x];
		localPosZ[threadIdx.x] = pos_z[tile + threadIdx.x];
		localmass[threadIdx.x] = mass[tile + threadIdx.x];

		__syncthreads();

		for (unsigned int index = 0; index < threads_num; index++) {
			if ((index + tile) == tid)
				continue;

			double3 other = { localPosX[index],
							localPosY[index],
							localPosZ[index] };
			double3 r = { other.x - body.x,
						other.y - body.y,
						other.z - body.z };

			double dis_squared = r.x * r.x + r.y * r.y + r.z * r.z;
			double one_over_dis_cube = 1.0 / pow(sqrt(dis_squared + epsilon_squared), 3);


			acl.x += localmass[index] * one_over_dis_cube * r.x;
			acl.y += localmass[index] * one_over_dis_cube * r.y;
			acl.z += localmass[index] * one_over_dis_cube * r.z;

		}
		__syncthreads();
	}

	//debug

	//if (tid == 4095) {
	//	double3 other = { localPosX[127],
	//						localPosY[127],
	//						localPosZ[127] };
	//	double3 r = { other.x - body.x,
	//				other.y - body.y,
	//				other.z - body.z };

	//	double dis_squared = r.x * r.x + r.y * r.y + r.z * r.z;
	//	double one_over_dis_cube = 1.0 / pow(sqrt(dis_squared + epsilon_squared), 3);
	//	//printf("counter: %d", counter);

	//	printf("x dis: %.6f\n", localPosX[127]);
	//	printf("y dis: %.6f\n", localPosY[127]);
	//	printf("z dis: %.6f\n", localPosZ[127]);
	//}


	acl_x[tid] = acl.x;
	acl_y[tid] = acl.y;
	acl_z[tid] = acl.z;


	vel_x[tid] += acl_x[tid] * dt;
	vel_y[tid] += acl_y[tid] * dt;
	vel_z[tid] += acl_z[tid] * dt;


	//pos_x[tid] += vel_x[tid] * dt;
	//pos_y[tid] += vel_y[tid] * dt;
	//pos_z[tid] += vel_z[tid] * dt;


}

__global__ void update_pos(double* pos_x, double* pos_y, double* pos_z,		////position array
	double* vel_x, double* vel_y, double* vel_z,
	const double dt)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	pos_x[tid] += vel_x[tid] * dt;
	pos_y[tid] += vel_y[tid] * dt;
	pos_z[tid] += vel_z[tid] * dt;

}

__host__ void Test_N_Body_Simulation()
{
	////initialize position, velocity, acceleration, and mass

	double* pos_x = new double[particle_n];
	double* pos_y = new double[particle_n];
	double* pos_z = new double[particle_n];
	////initialize particle positions as the cell centers on a background grid
	double dx = 1.0 / (double)grid_size;
	for (unsigned int k = 0; k < grid_size; k++) {
		for (unsigned int j = 0; j < grid_size; j++) {
			for (unsigned int i = 0; i < grid_size; i++) {
				unsigned int index = k * grid_size * grid_size + j * grid_size + i;
				pos_x[index] = dx * (double)i;
				pos_y[index] = dx * (double)j;
				pos_z[index] = dx * (double)k;
			}
		}
	}

	double* vel_x = new double[particle_n];
	memset(vel_x, 0x00, particle_n * sizeof(double));
	double* vel_y = new double[particle_n];
	memset(vel_y, 0x00, particle_n * sizeof(double));
	double* vel_z = new double[particle_n];
	memset(vel_z, 0x00, particle_n * sizeof(double));

	double* acl_x = new double[particle_n];
	memset(acl_x, 0x00, particle_n * sizeof(double));
	double* acl_y = new double[particle_n];
	memset(acl_y, 0x00, particle_n * sizeof(double));
	double* acl_z = new double[particle_n];
	memset(acl_z, 0x00, particle_n * sizeof(double));

	double* mass = new double[particle_n];
	for (int i = 0; i < particle_n; i++) {
		mass[i] = 100.0;
	}


	//////////////////////////////////////////////////////////////////////////
	////Default implementation: n-body simulation on CPU
	////Comment the CPU implementation out when you test large-scale examples
	auto cpu_start = chrono::system_clock::now();
	cout << "Total number of particles: " << particle_n << endl;
	cout << "Tracking the motion of particle " << particle_n / 2 << endl;


	/*for(int i=0;i<time_step_num;i++){
		N_Body_Simulation_CPU_Poorman(pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acl_x,acl_y,acl_z,mass,particle_n,dt,epsilon_squared);
		cout<<"pos on timestep "<<i<<": "<<pos_x[particle_n /2]<<", "<<pos_y[particle_n /2]<<", "<<pos_z[particle_n /2]<<endl;
	}*/


	auto cpu_end = chrono::system_clock::now();
	chrono::duration<double> cpu_time = cpu_end - cpu_start;
	cout << "CPU runtime: " << cpu_time.count() * 1000. << " ms." << endl;

	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	////Your implementation: n-body simulator on GPU

	double* pos_x_dev;
	double* pos_y_dev;
	double* pos_z_dev;
	double* vel_x_dev;
	double* vel_y_dev;
	double* vel_z_dev;
	double* acl_x_dev;
	double* acl_y_dev;
	double* acl_z_dev;

	double* mass_dev;

	cudaMalloc((void**)&mass_dev, particle_n * sizeof(double));
	cudaMemcpy(mass_dev, mass, particle_n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&pos_x_dev, particle_n * sizeof(double));
	cudaMemcpy(pos_x_dev, pos_x, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&pos_y_dev, particle_n * sizeof(double));
	cudaMemcpy(pos_y_dev, pos_y, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&pos_z_dev, particle_n * sizeof(double));
	cudaMemcpy(pos_z_dev, pos_z, particle_n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&vel_x_dev, particle_n * sizeof(double));
	cudaMemcpy(vel_x_dev, vel_x, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&vel_y_dev, particle_n * sizeof(double));
	cudaMemcpy(vel_y_dev, vel_y, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&vel_z_dev, particle_n * sizeof(double));
	cudaMemcpy(vel_z_dev, vel_z, particle_n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&acl_x_dev, particle_n * sizeof(double));
	cudaMemcpy(acl_x_dev, acl_x, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&acl_y_dev, particle_n * sizeof(double));
	cudaMemcpy(acl_y_dev, acl_y, particle_n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&acl_z_dev, particle_n * sizeof(double));
	cudaMemcpy(acl_z_dev, acl_z, particle_n * sizeof(double), cudaMemcpyHostToDevice);


	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time = 0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//////////////////////////////////////////////////////////////////////////
	////TODO 2: Your GPU functions are called here
	////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
	////The final positions should be stored in the same place as the CPU n-body function, i.e., pos_x, pos_y, pos_z
	////The correctness of your simulation will be evaluated by comparing the results (positions) with the results calculated by the default CPU implementations


	for (int i = 0; i < time_step_num; i++) {
		compute_acl << <dim3(block_num), dim3(threads_num) >> > (pos_x_dev, pos_y_dev, pos_z_dev,
			vel_x_dev, vel_y_dev, vel_z_dev,
			acl_x_dev, acl_y_dev, acl_z_dev,
			mass_dev, particle_n, dt, epsilon_squared);
		

		update_pos << <dim3(block_num), dim3(threads_num) >> > (pos_x_dev, pos_y_dev, pos_z_dev,
			vel_x_dev, vel_y_dev, vel_z_dev, dt);

		cudaMemcpy(pos_x, pos_x_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(pos_y, pos_y_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(pos_z, pos_z_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);

		cout << "pos on timestep " << i << ": " << pos_x[particle_n / 2] << ", " << pos_y[particle_n / 2] << ", " << pos_z[particle_n / 2] << endl;
	}




	//////////////////////////////////////////////////////////////////////////



	//cudaMemcpy(vel_x, vel_x_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(vel_y, vel_y_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(vel_z, vel_z_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);


	//cudaMemcpy(acl_x, acl_x_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(acl_y, acl_y_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(acl_z, acl_z_dev, particle_n * sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, start, end);
	printf("\nGPU runtime: %.4f ms\n", gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////

	out << "R0: " << pos_x[particle_n / 2] << " " << pos_y[particle_n / 2] << " " << pos_z[particle_n / 2] << endl;
	out << "T1: " << gpu_time << endl;
}

int main()
{
	if (name::team == "Team_X") {
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name = name::team + "_competition_2_nbody.dat";
	out.open(file_name.c_str());

	if (out.fail()) {
		printf("\ncannot open file %s to record results\n", file_name.c_str());
		return 0;
	}

	Test_N_Body_Simulation();

	return 0;
}