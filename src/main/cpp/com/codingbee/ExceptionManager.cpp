#include "ExceptionManager.h"
#include <string>
#include <iostream>
#include <unistd.h>
#include <mingw-w64-headers/crt/unistd.h>

using namespace std;

boolean ExceptionManager::terminationOverride = false;
float ExceptionManager::terminationDelay = 5000;

void ExceptionManager::processException(string message, boolean shouldTerminate){
    cerr << "Exception occured: " << message << "\n";
    if (shouldTerminate && !terminationOverride){
        cerr << "Program will be terminated and all memory will be free after the delay: " << terminationDelay << "ms \n";
        sleep(terminationDelay);
        terminate();
    }

    if(shouldTerminate && terminationOverride){
        cerr << "The programm considers the exception fatal, but the termination was prevented by the override.\n";
    }

    if(!shouldTerminate){
        cerr << "The programm did not consider this exception fatal and will not terminate.\n"
    }
}