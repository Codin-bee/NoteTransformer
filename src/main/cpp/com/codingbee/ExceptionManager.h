#ifndef EXCEPTIONMANAGER_H
#define EXCEPTIONMANAGER_H

#include <string>
#include <rpcndr.h>

class ExceptionManager{
    public:
    void processException(std::string message, boolean shouldTerminate);
    private:
    static boolean terminationOverride;
    static long terminationDelay;
};
#endif