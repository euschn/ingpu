// soa agent tuple

#include "agent.h"
#include <iostream>
#include <sstream>

using namespace soa;

// return a dummy tuple


// print an agent
__host__
std::ostream& operator<< (std::ostream& os, const soa::agent& a) {
		os << std::setiosflags(std::ios::left);
		name_type name = thrust::get<NAME>(a);
		if (name >= 'a' && name <= 'z') {
			os << std::setw(2) << thrust::get<ID>(a);
		}
		else {
			os << std::setw(2) << name;
		}
		os << std::resetiosflags(std::ios::left);

		return os;
}

// print an agent
__host__
std::ostream& operator<< (std::ostream& os, const soa::equation& a) {
		os << std::setiosflags(std::ios::left)
		<< get<0>(a)
		<< "="
		<< get<1>(a);
		os << std::resetiosflags(std::ios::left);

		return os;
}

void print_equation(const soa::agent& lhs, const soa::agent& rhs)
{
			std::string left_args, right_args;
			if (thrust::get<soa::ARITY>(lhs) == 0) {
				left_args = "";
			}
			else {
				
				std::stringstream out;
				out << "(" << get<soa::P1>(lhs);
				if (thrust::get<soa::ARITY>(lhs) > 1) {
					out << ", " << get<soa::P2>(lhs);
				}
				if (thrust::get<soa::ARITY>(lhs) > 2) {
					out << ", " << get<soa::P3>(lhs);
				}
				if (thrust::get<soa::ARITY>(lhs) > 3) {
					out << ", " << get<soa::P4>(lhs);
				}
				out << ")";

				left_args = out.str();
			}

			if (thrust::get<soa::ARITY>(rhs) == 0) {
				right_args = "   ";
			}
			else {
				std::stringstream out;
				out << "(" << get<soa::P1>(rhs);
				if (thrust::get<soa::ARITY>(rhs) > 1) {
					out << ", " << get<soa::P2>(rhs);
				}
				if (thrust::get<soa::ARITY>(rhs) > 2) {
					out << ", " << get<soa::P3>(rhs);
				}
				if (thrust::get<soa::ARITY>(rhs) > 3) {
					out << ", " << get<soa::P4>(rhs);
				}
				out << ")";
				right_args = out.str();
			}
			std::cout << lhs
				<< left_args 
				<< " = "
				<< rhs 
				<< right_args 
				<< std::setw(10) <<  "  (" 
				<< thrust::get<soa::ID>(lhs) << ", " << get<soa::ID>(rhs)  << ") \n";

}
