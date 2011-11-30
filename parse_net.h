// parsing agent structs from input or a file
// largely based on boost::spirit's "parsing into a struct tutorial"
// http://www.boost.org/doc/libs/1_47_0/libs/spirit/doc/html/spirit/qi/tutorials/employee___parsing_into_structs.html

#ifndef PARSENET_H_
#define PARSENET_H_

#include <boost/config/compiler/nvcc.hpp>
#include <vector>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost\fusion\include\adapt_struct.hpp>
#include <boost\fusion\include\io.hpp>
#include "agent_struct.h"
#include "soa/agent.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_map>

using namespace std::tr1;
using namespace std;
using namespace ingpu;


namespace parse_net {

	//namespace shortcuts
	namespace qi = boost::spirit::qi;
	namespace ascii = boost::spirit::ascii;

	//change this to char in the future
	typedef std::string agent_symbol;
	typedef std::string variable_symbol;

	//raw agent struct
	//reads symbols, but doesnt have correct agent ids yet
	struct raw_agent{
		agent_symbol symbol;
		std::vector<variable_symbol> ports;


		std::string to_string();
	};
}

	// We need to tell fusion about our raw_agent struct
	// to make it a first-class fusion citizen. This has to
	// be in global scope.
	BOOST_FUSION_ADAPT_STRUCT(
		parse_net::raw_agent,
		(parse_net::agent_symbol, symbol)
		(std::vector<parse_net::variable_symbol>, ports)
	)

namespace parse_net {
	// ----------------
	// raw agent parser
	// ----------------
	// TODO: parsing A() (parentheses with no ports) will yield an error!
	struct raw_agent_parser : qi::grammar<std::string::iterator, raw_agent(), ascii::space_type> 
	{
		raw_agent_parser() : raw_agent_parser::base_type(start)
		{
			using ascii::alpha;
			using ascii::alnum;

			start %=
				+(alnum)
				>> -(
					'('
					>> (+(alnum) % ',')
					>> ')'
					)
				;

		}

		qi::rule<std::string::iterator, raw_agent(), ascii::space_type> start;
	};

	bool parse_single(const std::string& filepath, std::vector<raw_agent>& raws);
	bool parse_all(const std::string& filepath, vector<vector<raw_agent>>& raws_vector);

	void build_agents(const std::vector<parse_net::raw_agent>& raws, std::vector<Agent>& agents);
	// soa version
	void build_agents(const std::vector<parse_net::raw_agent>& raws, 
		std::vector<soa::id_type>& ids, 
		std::vector<soa::name_type>& names, 
		std::vector<soa::arity_type>& arities, 
		std::vector<soa::port_type>& p1s, 
		std::vector<soa::port_type>& p2s, 
		std::vector<soa::port_type>& p3s, 
		std::vector<soa::port_type>& p4s);
}
#endif