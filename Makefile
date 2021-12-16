### begin MACROS
# define compiler
CC=nvcc

# compilation flags
CFLAGS= 

# define addition librariry (like −lm)
LIBS=

# define sources files to compile
SOURCES=main.cu

# define the name of the executable
EXEC=main
### end MACROS

### begin targets
build: $(SOURCES) $(EXEC)

$(EXEC): $(SOURCES)
	$(CC) $(CFLAGS) $< -o $@

### end targets
