#ifndef EXCEPTIONMANAGER_H
#define EXCEPTIONMANAGER_H

#include <string>

class ExceptionManager{
    public:
    static void processException(std::string message, bool shouldTerminate);
    private:
    static bool terminationOverride;
    static long terminationDelay;
};
#endif