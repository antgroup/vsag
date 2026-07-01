
//          Copyright Oliver Kowalke 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <boost/array.hpp>
#include <boost/assert.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/utility.hpp>

#include <boost/fiber/numa/pin_thread.hpp>
#include <boost/fiber/numa/topology.hpp>

void test_topology() {
    bool empty = boost::fibers::numa::topology().empty();
    BOOST_CHECK( ! empty);
}

void test_pin_self() {
    auto topo = boost::fibers::numa::topology();
    BOOST_CHECK( ! topo.empty() );
    BOOST_CHECK( ! topo[0].logical_cpus.empty() );
    boost::fibers::numa::pin_thread( * topo[0].logical_cpus.begin() );
}

void test_pin() {
    auto topo = boost::fibers::numa::topology();
    BOOST_CHECK( ! topo.empty() );
    BOOST_CHECK( ! topo[0].logical_cpus.empty() );
    std::thread t{ []{ std::this_thread::sleep_for( std::chrono::seconds( 1) ); } };
    boost::fibers::numa::pin_thread( * topo[0].logical_cpus.begin(), t.native_handle() );
    t.join();
}

boost::unit_test::test_suite * init_unit_test_suite( int, char* []) {
    boost::unit_test::test_suite * test =
        BOOST_TEST_SUITE("Boost.Fiber: numa topology test suite");

    test->add( BOOST_TEST_CASE( & test_topology) );
#if 0
    test->add( BOOST_TEST_CASE( & test_pin_self) );
    test->add( BOOST_TEST_CASE( & test_pin) );
#endif

    return test;
}
