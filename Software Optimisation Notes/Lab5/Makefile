# list all binaries in this next line
BINARIES = vol1
CCOPTS = -g -O3

all:	${BINARIES}

vol1:	vol1.c vol.h
	gcc ${CCOPTS} vol1.c -o vol1

test1:	vol1
	bash -c "time ./vol1"

# target to test all binaries
test:	test1 

gdb1:	vol1
	gdb vol1

clean:	
	rm ${BINARIES} || true


