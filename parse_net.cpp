#include "parse_net.h"
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/foreach.hpp>

#include <string>
#include <iostream>
#include <fstream>

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;


using namespace ingpu;

bool parse_net::parse_single(const std::string& filepath, std::vector<raw_agent>& raws)
{
	using qi::phrase_parse;
	using qi::_1;
	using ascii::space;
	using ascii::alpha;
	using ascii::alnum;

	std::ifstream fp_in;
	fp_in.open(filepath, std::ios::in);
	if (!fp_in.is_open()) {
		std::cout << "error opening file " << filepath << std::endl;
		return false;
	}

	std::string str;
	bool result = false;
	while ( getline(fp_in, str) )
	{
		if (str.empty())
			continue;
		// ignore commented line
		if (str[0] == '/' || str[1] == '/')
			continue;

		//raw_agent parsing
		// ----------------
		//std::vector<parse_net::raw_agent> ra;
		parse_net::raw_agent_parser p;
		result = phrase_parse(str.begin(), str.end(), 
			(
				(p >> '=' >> p) % ','
			)
			, space, raws);
		if (result) {
			std::cout << "----------------------------\nparsed string: " << str << std::endl;
			break;
		}
	}

	return result;
}


bool parse_net::parse_all(const std::string& filepath, vector<vector<raw_agent>>& raws_vector)
{
	using qi::phrase_parse;
	using qi::_1;
	using ascii::space;
	using ascii::alpha;
	using ascii::alnum;

	std::ifstream fp_in;
	fp_in.open(filepath, std::ios::in);
	if (!fp_in.is_open()) {
		std::cout << "error opening file " << filepath << std::endl;
		return false;
	}

	std::string str;
	bool result = false;
	while ( getline(fp_in, str) )
	{
		if (str.empty())
			continue;
		// ignore commented line
		if (str[0] == '/' || str[1] == '/')
			continue;

		//raw_agent parsing
		// ----------------
		std::vector<parse_net::raw_agent> raws;
		parse_net::raw_agent_parser p;
		result = phrase_parse(str.begin(), str.end(), 
			(
				(p >> '=' >> p) % ','
			)
			, space, raws);
		if (result) {
			raws_vector.push_back(raws);
			std::cout << "----------------------------\nparsed string: " << str << std::endl;
		}
	}

	return result;
}

std::string parse_net::raw_agent::to_string()
{
	std::string result = "";
	result += this->symbol;
	if (this->ports.size() > 0)
	{
		result += '(';
		BOOST_FOREACH(variable_symbol port, this->ports)
		{
			result += port;
			result += " ";
		}
		result += ')';
	}

	return result;
}

void parse_net::build_agents(const std::vector<parse_net::raw_agent>& raws, std::vector<Agent>& agents)
{
	typedef parse_net::raw_agent raw;
	typedef std::unordered_map<std::string, int> id_map;
	id_map map;

	int id_counter=1;
	BOOST_FOREACH(raw ragent, raws)
	{
		int raw_port_size = ragent.ports.size();
		if (raw_port_size > MAX_PORTS) {
			std::cerr << "error: agent " << ragent.symbol << "has too many ports, using only the first " << MAX_PORTS << " ports\n";
		}
		Agent a(0, ragent.symbol.at(0), std::min(raw_port_size, MAX_PORTS));
		id_map::iterator it = map.find(ragent.symbol);
		// if symbol has not occured before or is upper case
		if ( ( it == map.end() ) || (a.name >= 'A' && a.name <= 'Z') )
		{
			//assign a new id and add it to the map
			a.id = id_counter;
			if ( it == map.end() ) {
				// only new variables get added to the map
				map[ragent.symbol] = id_counter;
			}
			id_counter++;
		}
		else {
			a.id = it->second;
		}
		// assing ids to the ports the same way
		for (int i=0; i < a.arity; ++i) {
			std::string current_symbol = ragent.ports[i];
			id_map::iterator it = map.find( current_symbol );
			// if symbol has not occured before or is upper case
			if ( ( it == map.end() ) || ( current_symbol.at(0) >= 'A' &&  current_symbol.at(0) <= 'Z') )
			{
				//assign a new id and add it to the map
				a.ports[i] = id_counter;
				if ( it == map.end() ) {
					// only new variables get added to the map
					map[ current_symbol ] = id_counter;
				}
				id_counter++;
			}
			else {
				a.ports[i] = it->second;
			}
		}

		//we are done building, add the agent to the output vector
		agents.push_back(a);
	}
	

}

//build a set of soa agents from parsed raw agents
void parse_net::build_agents(const std::vector<parse_net::raw_agent>& raws, 
	std::vector<soa::id_type>& ids, 
	std::vector<soa::name_type>& names, 
	std::vector<soa::arity_type>& arities, 
	std::vector<soa::port_type>& p1s, 
	std::vector<soa::port_type>& p2s, 
	std::vector<soa::port_type>& p3s, 
	std::vector<soa::port_type>& p4s) 
{
	
	typedef parse_net::raw_agent raw;
	typedef std::unordered_map<std::string, int> id_map;
	id_map map;

	int id_counter=1;
	BOOST_FOREACH(raw ragent, raws)
	{
		int raw_port_size = ragent.ports.size();
		if (raw_port_size > MAX_PORTS) {
			std::cerr << "error: agent " << ragent.symbol << "has too many ports, using only the first " << MAX_PORTS << " ports\n";
		}
		//soa::agent a = soa::make_agent(0, ragent.symbol, std::min(raw_port_size, MAX_PORTS));
		soa::id_type id = 0;
		soa::name_type name = ragent.symbol.at(0);
		soa::arity_type arity = std::min(raw_port_size, MAX_PORTS);
		id_map::iterator it = map.find(ragent.symbol);
		// if symbol has not occured before or is upper case
		if ( ( it == map.end() ) || (name >= 'A' && name <= 'Z') )
		{
			//assign a new id and add it to the map
			id = id_counter;
			if ( it == map.end() ) {
				// only new variables get added to the map
				map[ragent.symbol] = id_counter;
			}
			id_counter++;
		}
		else {
			id = it->second;
		}
		// assing ids to the ports the same way
		int final_ports[MAX_PORTS];
		for (int i=0; i < arity; ++i) {
			std::string current_symbol = ragent.ports[i];
			id_map::iterator it = map.find( current_symbol );
			// if symbol has not occured before or is upper case
			if ( ( it == map.end() ) || ( current_symbol.at(0) >= 'A' &&  current_symbol.at(0) <= 'Z') )
			{
				//assign a new id and add it to the map
				final_ports[i] = id_counter;
				if ( it == map.end() ) {
					// only new variables get added to the map
					map[ current_symbol ] = id_counter;
				}
				id_counter++;
			}
			else {
				final_ports[i] = it->second;
			}
		}

		//we are done building, add the agent to the output vector
		ids.push_back(id);
		names.push_back(name);
		arities.push_back(arity);
		p1s.push_back(final_ports[0]);
		p2s.push_back(final_ports[1]);
		p3s.push_back(final_ports[2]);
		p4s.push_back(final_ports[3]);
	}
}