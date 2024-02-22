all:
	nvcc -o main main.cu mm0.cu mm1.cu -lineinfo -arch=sm_80

clean:
	rm main
