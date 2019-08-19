BINARIES = vol_simd
CCOPTS = -g -O3

all:	${BINARIES}

vol_simd:	vol_simd.c vol.h
		gcc ${CCOPTS} vol_simd.c -o vol_simd

test:		vol_simd
		bash -c "time ./vol_simd"

gdb:	vol_simd
	gdb vol_simd

clean:	
	rm ${BINARIES} || true


