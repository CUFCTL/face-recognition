# 1 "<stdin>"
# 1 "<built-in>"
# 1 "<command line>"
# 12 "<command line>"
# 1 "pca_hw.c" 1

# 1 "C:/Impulse/CoDeveloper3/MinGW/include/windows.h" 1 3
# 17 "C:/Impulse/CoDeveloper3/MinGW/include/windows.h" 3
# 44 "C:/Impulse/CoDeveloper3/MinGW/include/windows.h" 3
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/winresrc.h" 1 3
# 5 "C:/Impulse/CoDeveloper3/MinGW/include/winresrc.h" 3



# 1 "C:/Impulse/CoDeveloper3/MinGW/include/winuser.h" 1 3
# 5 "C:/Impulse/CoDeveloper3/MinGW/include/winuser.h" 3
# 9 "C:/Impulse/CoDeveloper3/MinGW/include/winresrc.h" 2 3
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/winnt.h" 1 3
# 5 "C:/Impulse/CoDeveloper3/MinGW/include/winnt.h" 3
# 34 "C:/Impulse/CoDeveloper3/MinGW/include/winnt.h" 3
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/winerror.h" 1 3
# 5 "C:/Impulse/CoDeveloper3/MinGW/include/winerror.h" 3
# 35 "C:/Impulse/CoDeveloper3/MinGW/include/winnt.h" 2 3
# 158 "C:/Impulse/CoDeveloper3/MinGW/include/winnt.h" 3
typedef unsigned char FCHAR;
typedef unsigned short FSHORT;
typedef unsigned int FLONG;


# 1 "C:/Impulse/CoDeveloper3/MinGW/include/basetsd.h" 1 3
# 5 "C:/Impulse/CoDeveloper3/MinGW/include/basetsd.h" 3
# 164 "C:/Impulse/CoDeveloper3/MinGW/include/winnt.h" 2 3
# 10 "C:/Impulse/CoDeveloper3/MinGW/include/winresrc.h" 2 3
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/winver.h" 1 3
# 5 "C:/Impulse/CoDeveloper3/MinGW/include/winver.h" 3
# 11 "C:/Impulse/CoDeveloper3/MinGW/include/winresrc.h" 2 3
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/dde.h" 1 3
# 5 "C:/Impulse/CoDeveloper3/MinGW/include/dde.h" 3
# 12 "C:/Impulse/CoDeveloper3/MinGW/include/winresrc.h" 2 3
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/dlgs.h" 1 3
# 5 "C:/Impulse/CoDeveloper3/MinGW/include/dlgs.h" 3
# 13 "C:/Impulse/CoDeveloper3/MinGW/include/winresrc.h" 2 3
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/commctrl.h" 1 3
# 5 "C:/Impulse/CoDeveloper3/MinGW/include/commctrl.h" 3


# 1 "C:/Impulse/CoDeveloper3/MinGW/include/prsht.h" 1 3
# 5 "C:/Impulse/CoDeveloper3/MinGW/include/prsht.h" 3
# 8 "C:/Impulse/CoDeveloper3/MinGW/include/commctrl.h" 2 3
# 14 "C:/Impulse/CoDeveloper3/MinGW/include/winresrc.h" 2 3
# 45 "C:/Impulse/CoDeveloper3/MinGW/include/windows.h" 2 3
# 3 "pca_hw.c" 2

# 1 "C:/Impulse/CoDeveloper3/MinGW/include/stdio.h" 1 3
# 35 "C:/Impulse/CoDeveloper3/MinGW/include/stdio.h" 3
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/_mingw.h" 1 3
# 36 "C:/Impulse/CoDeveloper3/MinGW/include/stdio.h" 2 3
# 5 "pca_hw.c" 2
# 1 "C:/Impulse/CoDeveloper3/Include/co.h" 1
# 27 "C:/Impulse/CoDeveloper3/Include/co.h"
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/fcntl.h" 1 3
# 39 "C:/Impulse/CoDeveloper3/MinGW/include/fcntl.h" 3
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/io.h" 1 3
# 43 "C:/Impulse/CoDeveloper3/MinGW/include/io.h" 3
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/sys/types.h" 1 3
# 44 "C:/Impulse/CoDeveloper3/MinGW/include/io.h" 2 3
# 40 "C:/Impulse/CoDeveloper3/MinGW/include/fcntl.h" 2 3
# 28 "C:/Impulse/CoDeveloper3/Include/co.h" 2
# 1 "C:/Impulse/CoDeveloper3/Include/co_types.h" 1
# 25 "C:/Impulse/CoDeveloper3/Include/co_types.h"
typedef enum {co_err_none = 0, co_err_eos, co_err_invalid_arg, co_err_already_open, co_err_not_open, co_err_unavail, co_err_unknown} co_error;

typedef char int8;
typedef unsigned char uint8;
typedef short int16;
typedef unsigned short uint16;
typedef int int32;
typedef unsigned int uint32;




typedef long long int64;
typedef unsigned long long uint64;


typedef enum {co_int_sort = 1, co_uint_sort, co_float_sort} co_sort;

struct co_type_s;
typedef struct co_type_s *co_type;

typedef enum {co_loc, co_kind} co_attribute;

struct co_arch_s;
typedef struct co_arch_s *co_architecture;

typedef void *co_parameter;

typedef void (*co_function)();
typedef void (*co_config_function)(void *);

struct co_process_s;
typedef struct co_process_s *co_process;

struct co_memory_s;
typedef struct co_memory_s *co_memory;

struct co_signal_s;
typedef struct co_signal_s *co_signal;

struct co_register_s;
typedef struct co_register_s *co_register;

struct co_semaphore_s;
typedef struct co_semaphore_s *co_semaphore;

typedef void *co_port;
typedef enum {co_input=0, co_output} co_port_mode;
# 86 "C:/Impulse/CoDeveloper3/Include/co_types.h"
typedef struct co_stream_s {
        char *name;
        unsigned int flags;
        co_type type;

        unsigned int impl_size;
        unsigned int size;
        union {
                uint8 *fifo8;
                uint16 *fifo16;
                uint32 *fifo32;
                uint64 *fifo64;
        } u;
        uint8 *storage, *in_pos, *end;
        uint8 *ctrl_storage, *ctrl_in_pos, *ctrl_fifo, *ctrl_end;
        unsigned int fill_count;
        unsigned int reader,writer;
        void *rgate,*wgate;
        void *mutex;
        void *waveform;





} co_stream_t;
typedef struct co_stream_s *co_stream;
# 29 "C:/Impulse/CoDeveloper3/Include/co.h" 2
# 1 "C:/Impulse/CoDeveloper3/Include/co_import.h" 1
# 30 "C:/Impulse/CoDeveloper3/Include/co.h" 2




extern co_architecture co_architecture_create(const char *name, const char *arch, co_config_function configure, void *arg);
extern co_process co_process_create(const char *name, co_function run, int argc, ...);
extern co_error co_process_config(co_process process, co_attribute attribute, const char *value);
extern co_error co_array_config(void * buffer, co_attribute attribute, const char * value);
extern void co_execute(co_architecture architecture);



extern co_type co_type_create(co_sort sort, unsigned int width);
# 63 "C:/Impulse/CoDeveloper3/Include/co.h"
extern co_stream co_stream_create(const char *name, co_type type, int numelements);
extern co_error co_stream_config(co_stream stream, co_attribute attribute, const char *val);
extern co_error co_stream_open(co_stream stream, int mode, co_type type);
extern co_error co_stream_close(co_stream stream);
extern co_error co_stream_read(co_stream stream, void *buffer, long unsigned int buffersize);
extern int co_stream_read_nb(co_stream stream, void *buffer, long unsigned int buffersize);
extern co_error co_stream_write(co_stream stream, const void *buffer, long unsigned int buffersize);
extern int co_stream_write_nb(co_stream stream, const void *buffer, long unsigned int buffersize);
extern int co_stream_eos(co_stream stream);
extern co_error co_stream_free(co_stream stream);
# 96 "C:/Impulse/CoDeveloper3/Include/co.h"
extern co_signal co_signal_create(const char *name);
extern co_signal co_signal_create_ex(const char *name, co_type type);
extern co_error co_signal_wait(co_signal signal, int32 *ip);
extern co_error co_signal_post(co_signal signal, int32 value);
extern co_error co_signal_free(co_signal signal);



extern co_register co_register_create(const char *name, co_type type);
extern co_error co_register_write(co_register reg, const void *buffer, long unsigned int buffersize);
extern co_error co_register_read(co_register reg, void *buffer, long unsigned int buffersize);
extern void co_register_put(co_register reg, int32 value);
extern int32 co_register_get(co_register reg);
extern co_error co_register_free(co_register reg);


extern co_semaphore co_semaphore_create(const char *name, int init, int max);
extern co_error co_semaphore_wait(co_semaphore sema);
extern co_error co_semaphore_release(co_semaphore sema);
extern co_error co_semaphore_free(co_semaphore sema);



extern co_memory co_memory_create(const char *name, const char *loc, long unsigned int size);
extern void co_memory_writeblock(co_memory mem, unsigned int offset, void *buf, long unsigned int buffersize);
extern void co_memory_readblock(co_memory mem, unsigned int offset, void *buf, long unsigned int buffersize);
extern void *co_memory_ptr(co_memory mem);
extern co_error co_memory_free(co_memory mem);
extern void co_memory_set(co_memory m, void* buf, long unsigned int bufsize);
# 134 "C:/Impulse/CoDeveloper3/Include/co.h"
extern co_port co_port_create(const char *name, co_port_mode mode, void *io_object);


# 1 "C:/Impulse/CoDeveloper3/Include/co_if_sim.h" 1
# 138 "C:/Impulse/CoDeveloper3/Include/co.h" 2


# 1 "C:/Impulse/CoDeveloper3/Include/co_math.h" 1
# 24 "C:/Impulse/CoDeveloper3/Include/co_math.h"
typedef int8 co_int1;
typedef int8 co_int2;
typedef int8 co_int3;
typedef int8 co_int4;
typedef int8 co_int5;
typedef int8 co_int6;
typedef int8 co_int7;
typedef int8 co_int8;
typedef int16 co_int9;
typedef int16 co_int10;
typedef int16 co_int11;
typedef int16 co_int12;
typedef int16 co_int13;
typedef int16 co_int14;
typedef int16 co_int15;
typedef int16 co_int16;
typedef int32 co_int17;
typedef int32 co_int18;
typedef int32 co_int19;
typedef int32 co_int20;
typedef int32 co_int21;
typedef int32 co_int22;
typedef int32 co_int23;
typedef int32 co_int24;
typedef int32 co_int25;
typedef int32 co_int26;
typedef int32 co_int27;
typedef int32 co_int28;
typedef int32 co_int29;
typedef int32 co_int30;
typedef int32 co_int31;
typedef int32 co_int32;
typedef int64 co_int33;
typedef int64 co_int34;
typedef int64 co_int35;
typedef int64 co_int36;
typedef int64 co_int37;
typedef int64 co_int38;
typedef int64 co_int39;
typedef int64 co_int40;
typedef int64 co_int41;
typedef int64 co_int42;
typedef int64 co_int43;
typedef int64 co_int44;
typedef int64 co_int45;
typedef int64 co_int46;
typedef int64 co_int47;
typedef int64 co_int48;
typedef int64 co_int49;
typedef int64 co_int50;
typedef int64 co_int51;
typedef int64 co_int52;
typedef int64 co_int53;
typedef int64 co_int54;
typedef int64 co_int55;
typedef int64 co_int56;
typedef int64 co_int57;
typedef int64 co_int58;
typedef int64 co_int59;
typedef int64 co_int60;
typedef int64 co_int61;
typedef int64 co_int62;
typedef int64 co_int63;
typedef int64 co_int64;
typedef int32 co_int128[4];
typedef int32 co_int256[8];

typedef uint8 co_uint1;
typedef uint8 co_uint2;
typedef uint8 co_uint3;
typedef uint8 co_uint4;
typedef uint8 co_uint5;
typedef uint8 co_uint6;
typedef uint8 co_uint7;
typedef uint8 co_uint8;
typedef uint16 co_uint9;
typedef uint16 co_uint10;
typedef uint16 co_uint11;
typedef uint16 co_uint12;
typedef uint16 co_uint13;
typedef uint16 co_uint14;
typedef uint16 co_uint15;
typedef uint16 co_uint16;
typedef uint32 co_uint17;
typedef uint32 co_uint18;
typedef uint32 co_uint19;
typedef uint32 co_uint20;
typedef uint32 co_uint21;
typedef uint32 co_uint22;
typedef uint32 co_uint23;
typedef uint32 co_uint24;
typedef uint32 co_uint25;
typedef uint32 co_uint26;
typedef uint32 co_uint27;
typedef uint32 co_uint28;
typedef uint32 co_uint29;
typedef uint32 co_uint30;
typedef uint32 co_uint31;
typedef uint32 co_uint32;
typedef uint64 co_uint33;
typedef uint64 co_uint34;
typedef uint64 co_uint35;
typedef uint64 co_uint36;
typedef uint64 co_uint37;
typedef uint64 co_uint38;
typedef uint64 co_uint39;
typedef uint64 co_uint40;
typedef uint64 co_uint41;
typedef uint64 co_uint42;
typedef uint64 co_uint43;
typedef uint64 co_uint44;
typedef uint64 co_uint45;
typedef uint64 co_uint46;
typedef uint64 co_uint47;
typedef uint64 co_uint48;
typedef uint64 co_uint49;
typedef uint64 co_uint50;
typedef uint64 co_uint51;
typedef uint64 co_uint52;
typedef uint64 co_uint53;
typedef uint64 co_uint54;
typedef uint64 co_uint55;
typedef uint64 co_uint56;
typedef uint64 co_uint57;
typedef uint64 co_uint58;
typedef uint64 co_uint59;
typedef uint64 co_uint60;
typedef uint64 co_uint61;
typedef uint64 co_uint62;
typedef uint64 co_uint63;
typedef uint64 co_uint64;
typedef uint32 co_uint128[4];
typedef uint32 co_uint256[8];
# 717 "C:/Impulse/CoDeveloper3/Include/co_math.h"
extern float to_float(uint32 i);
extern double to_double(uint64 i);
extern uint32 float_bits(float f);
extern uint64 double_bits(double f);
# 748 "C:/Impulse/CoDeveloper3/Include/co_math.h"
extern double sqrt (double);
extern double fabs (double);

extern float sqrtf (float);
extern float fabsf (float);
# 141 "C:/Impulse/CoDeveloper3/Include/co.h" 2
# 6 "pca_hw.c" 2




# 1 "C:/Impulse/CoDeveloper3/Include/cosim_log.h" 1
# 24 "C:/Impulse/CoDeveloper3/Include/cosim_log.h"
# 1 "C:/Impulse/CoDeveloper3/MinGW/include/stdarg.h" 1 3
# 25 "C:/Impulse/CoDeveloper3/Include/cosim_log.h" 2



typedef struct cosim_logwindow_s* cosim_logwindow;
typedef struct cosim_logstream_s* cosim_logstream;




 int cosim_logwindow_init();



 cosim_logstream cosim_logstream_create(const char * name);
 cosim_logwindow cosim_logwindow_create(const char * name);




 int cosim_logstream_write(cosim_logstream log, const char * msg);
 int cosim_logwindow_write(cosim_logwindow log, const char * msg);


 int cosim_logstream_fwrite(cosim_logstream log, const char * format, ...);
 int cosim_logwindow_fwrite(cosim_logwindow log, const char * format, ...);


 void cosim_logstream_free(cosim_logstream log);
 void cosim_logwindow_free(cosim_logwindow log);
# 11 "pca_hw.c" 2



extern void Consumer(co_stream input_stream);
extern void Producer(co_stream output_stream, co_parameter iparam);

void Recognition(co_stream input_stream, co_stream output_stream)
{
        int s;

       
       

        do {
               
                co_stream_open(input_stream, 0, co_type_create(co_int_sort,32));
               
                co_stream_open(output_stream, 1, co_type_create(co_int_sort,32));

                while (co_stream_read(input_stream, &s, sizeof(int)) == co_err_none) {

                       
                       




                        co_stream_write(output_stream, &s,sizeof(int));
                }

                co_stream_close(input_stream);
                co_stream_close(output_stream);

               
        } while (1);
}


void config_recognition(void *arg)
{
        int iterations = (int) arg;
        co_stream s1,s2;
        co_process producer, consumer;
        co_process recognition;

       

        s1=co_stream_create("Stream1",co_type_create(co_int_sort,32),5);
        s2=co_stream_create("Stream2",co_type_create(co_int_sort,32),5);

        producer=co_process_create("Producer",(co_function) Producer, 2, s1, iterations);
        recognition=co_process_create("Recognition",(co_function) Recognition, 2, s1, s2);
        consumer=co_process_create("Consumer",(co_function) Consumer, 1, s2);

        co_process_config(recognition, co_loc, "PE0");
}

co_architecture co_initialize(int iterations)
{
        return(co_architecture_create("RecognitionArch","generic",config_recognition,(void *)iterations));
}
# 13 "<command line>" 2
# 1 "<stdin>"
