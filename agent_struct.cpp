#include "agent_struct.h"


// print an agent
__host__
std::ostream& operator<< (std::ostream& os, const ingpu::Agent& a) {
		os << std::setiosflags(std::ios::left);
		if (a.name >= 'a' && a.name <= 'z') {
			os << std::setw(2) << a.id;
		}
		else {
			os << std::setw(2) << a.name;
		}
		os << std::resetiosflags(std::ios::left);

		return os;
}