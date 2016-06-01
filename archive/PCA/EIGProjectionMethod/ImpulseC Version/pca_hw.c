#ifdef WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include "co.h"

#define MONITOR

#ifdef MONITOR
#include "cosim_log.h"
#endif

// Software processes defined in pca_sw.c
extern void Consumer(co_stream input_stream);
extern void Producer(co_stream output_stream, co_parameter iparam);

void Recognition(co_stream input_stream, co_stream output_stream)
{
	int s;

	IF_SIM(cosim_logwindow log;)
	IF_SIM(log = cosim_logwindow_create("Recognition");)	

	do {
		IF_SIM(cosim_logwindow_write(log, "Recognition opening stream: input_stream\n");)
		co_stream_open(input_stream, O_RDONLY, INT_TYPE(32));
		IF_SIM(cosim_logwindow_write(log, "Recognition opening stream: output_stream\n");)
		co_stream_open(output_stream, O_WRONLY, INT_TYPE(32));

		while (co_stream_read(input_stream, &s, sizeof(int)) == co_err_none) {
			// #pragma CO PIPELINE
			IF_SIM(cosim_logwindow_fwrite(log, "Recognition read %d from stream: input_stream\n", s);)
			IF_SIM(printf("Recognition read %d from stream: input_stream\n", s );)
			
			// operate on the data

			// write it back out
			co_stream_write(output_stream, &s,sizeof(int));
		}

		co_stream_close(input_stream);
		co_stream_close(output_stream);

		IF_SIM(break;)	
	} while (1);
}

#define BUFSIZE 5
void config_recognition(void *arg)
{
	int iterations = (int) arg;
	co_stream s1,s2;
	co_process producer, consumer;
	co_process recognition;

	IF_SIM(cosim_logwindow_init();)

	s1=co_stream_create("Stream1",INT_TYPE(32),BUFSIZE); // change stream to appropriate type here too
	s2=co_stream_create("Stream2",INT_TYPE(32),BUFSIZE);

	producer=co_process_create("Producer",(co_function) Producer, 2, s1, iterations);
	recognition=co_process_create("Recognition",(co_function) Recognition, 2, s1, s2);
	consumer=co_process_create("Consumer",(co_function) Consumer, 1, s2);

	co_process_config(recognition, co_loc, "PE0");  // assign recognition to hardware
}

co_architecture co_initialize(int iterations)
{
	return(co_architecture_create("RecognitionArch","generic",config_recognition,(void *)iterations));
}
