
//          Copyright Oliver Kowalke 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "boost/fiber/numa/pin_thread.hpp"

extern "C" {
#include <errno.h>
#include <sys/param.h>
#include <sys/cpuset.h>
#include <sys/thread.h>
}

#include <system_error>

#ifdef BOOST_HAS_ABI_HEADERS
# include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace fibers {
namespace numa {

BOOST_FIBERS_DECL
void pin_thread( std::uint32_t cpuid) {
    pin_thread( cpuid, ::thr_self() );
}

BOOST_FIBERS_DECL
void pin_thread( std::uint32_t cpuid, std::thread::native_handle_type h) {
	cpuset_t mask;
	CPU_ZERO( & mask);
	CPU_SET( cpuid, & mask);
    if ( BOOST_UNLIKELY( 0 != ::cpuset_setaffinity( CPU_LEVEL_WHICH, CPU_WHICH_TID, h, sizeof( mask), & mask) ) ) {
        throw std::system_error(
                std::error_code( errno, std::system_category() ),
                "::cpuset_setaffinity() failed");
    }
}

}}}

#ifdef BOOST_HAS_ABI_HEADERS
# include BOOST_ABI_SUFFIX
#endif
