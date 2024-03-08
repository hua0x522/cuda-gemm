all:
	nvcc -I . -g -o main main.cu mm0.cu mm9.cu -lineinfo -arch=sm_80

clean:
	rm main

prof:
	ncu --section regex:. -f --import-source yes -o main ./main 2048 2048 2048
