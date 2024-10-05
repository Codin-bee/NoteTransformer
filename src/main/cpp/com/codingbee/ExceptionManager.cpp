#include "ExceptionManager.h"
#include "FileUtils.h"
#include <string>
#include <iostream>
#include <unistd.h>

using namespace std;

bool ExceptionManager::terminationOverride = false;
long ExceptionManager::terminationDelay = 5000;

void ExceptionManager::processException(string message, bool shouldTerminate){
    cerr << "Exception occured: " << message << "\n";
    if (shouldTerminate && !terminationOverride){
        cerr << "Program will be terminated and all memory will be free after the delay: " << terminationDelay << "ms \n";
        sleep(terminationDelay);
        FileUtils::callRegisteredDestructors();
        terminate();
    }

    if(shouldTerminate && terminationOverride){
        cerr << "The programm considers the exception fatal, but the termination was prevented by the termination override.\n";
    }

    if(!shouldTerminate){
        cerr << "The programm did not consider this exception fatal and will not terminate.\n";
    }
}