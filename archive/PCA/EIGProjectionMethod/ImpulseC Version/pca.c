#ifdef WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include "co.h"

extern co_architecture co_initialize(int iterations);

int main(int argc, char *argv[]) {
	int c;
	int iterations = 2;
	co_architecture my_arch;

	my_arch = co_initialize(iterations);
	co_execute(my_arch);

	printf("ImpulsePCA done. Press Enter to continue...\n");
	c = getc(stdin);

	return(0);
}
