CXX = g++
CCFLAGS = -std=c++2a -g -Wall -Wextra -Wvla -Werror -Wno-error=unknown-pragmas \
		-Wno-error=unused-but-set-variable -Wno-error=unused-local-typedefs \
		-Wno-error=unused-function -Wno-error=unused-label -Wno-error=unused-value \
		-Wno-error=unused-variable -Wno-error=unused-parameter -Wno-error=unused-but-set-parameter \
		-Wno-psabi -march=native

.cc.o:
	$(CXX) $(CCFLAGS) -c $<

all: main

main: main.o
	$(CXX) $(LDFLAGS) -o out main.o


render: main
	./out > image.ppm
	
test: render
	gwenview image.ppm

clean:
	rm -f *.o
	rm -f out
