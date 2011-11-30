
#include "parse_net.h"
#include "kernel.h"
#include "config_manager.h"
#include <stopwatch.h>
#include <stopwatch_win.h>
#include <conio.h>
#include <thrust/tuple.h>
#include "soa/kernel_soa.h"
#include <fstream>

using namespace ingpu;
using namespace std;
using namespace thrust;

void eval_single(thrust::host_vector<ingpu::Agent>& in_lhs_host, thrust::host_vector<ingpu::Agent>& in_rhs_host)
{
	//-------
	//options
	bool print_final_net = config_manager::get_bool("print_final");
	bool verbose = config_manager::get_bool("verbose");
	bool pause = config_manager::get_bool("pause");
	const int ruleset = config_manager::get_int("ruleset");
	//partition or regular
	bool partition = config_manager::get_bool("partition");

	//timer
	const unsigned int mywatch = StopWatch::create();
	StopWatch::get(mywatch).start();

	/* ----------------
	// interaction loop
	   ----------------  */
	//arbitrary_transformation_example();
	result_info rinfo;
	//if (partition) {
	//	rinfo = 
	//	partition_interaction_step(ruleset, in_lhs_host, in_rhs_host, verbose, print_final_net, pause);
	//}
	//else  {
		rinfo = 
		test_interaction_step(ruleset, in_lhs_host, in_rhs_host, verbose, print_final_net, pause);
	//}
		//test_interaction_step(in_lhs_host, in_rhs_host, verbose, ins_type, print_final_net, pause);
	//	soa::interaction_loop( l_ids, l_names, l_arities, l_p1s, l_p2s, l_p3s, l_p4s,r_ids, r_names, r_arities, r_p1s, r_p2s, r_p3s, r_p4s, verbose, print_final_net);

	StopWatch::get(mywatch).stop();

	// stop and destroy timer
	float time_elapsed = StopWatch::get(mywatch).getTime() / 1000.0;
	printf ("elapsed time %.5f seconds.\n------------------\n", time_elapsed);
	StopWatch::destroy(mywatch);

	std::cout << "total loops: " << rinfo.num_loops << std::endl;
	std::cout << "total interactions: " << rinfo.total_interactions << " ( " << ( static_cast<float>(rinfo.total_interactions / time_elapsed)) << " / sec )"  << std::endl;
	const float avg_interactions = static_cast<float>(rinfo.total_interactions) / static_cast<float>(rinfo.num_loops);
	std::cout << "average interactions per loop: " << avg_interactions << std::endl;

	std::cout << "result (number of S symbols): " << rinfo.result << std::endl;
	std::cout << "\n----------------------------\n";

	//write stuff to file:
	if (config_manager::get_bool("file_output")) {
		std::ofstream outfile;
		outfile.open("out.csv", ios::app);
		outfile 
			<< time_elapsed << ";" 
			<< rinfo.num_loops << ";"
			<< rinfo.total_interactions << ";"
			<< avg_interactions << ";"
			<< (rinfo.total_interactions / time_elapsed) << ";"
			<< rinfo.com_time << ";"
			<< (rinfo.com_time / time_elapsed) << ";"
			<< rinfo.result
			<< "\n";
		outfile.close();
	}
}

void eval_all(vector < thrust::host_vector<ingpu::Agent> >& in_lhs_vector, vector< thrust::host_vector<ingpu::Agent> >& in_rhs_vector)
{
	size_t num_inputs = in_lhs_vector.size();
	std::cout << "evaluating " << num_inputs << " inputs\n";
	// file i/o
	if (config_manager::get_bool("file_output")) {
		std::ofstream outfile;
		outfile.open("out.csv");
		outfile << "elapsed time;loops;interactions; avg interactions/loop; interactions/sec; com time; com time %;result \n";
		outfile.close();
	}
	const bool pause_between_inputs = config_manager::get_bool("pause_between_inputs");
	for (unsigned int i=0; i< num_inputs; ++i) {
		eval_single(in_lhs_vector[i], in_rhs_vector[i]);
		if (pause_between_inputs) {
			std::cout << "press any key to continue\n";
			getch();
		}
	}
}

void eval_single_soa(host_vector<soa::id_type>& l_ids, host_vector<soa::name_type>& l_names, 
	host_vector<soa::arity_type>& l_arities, 
	host_vector<soa::port_type>& l_p1s, 
	host_vector<soa::id_type>& l_p2s, 
	host_vector<soa::id_type>& l_p3s, 
	host_vector<soa::id_type>& l_p4s,
	host_vector<soa::id_type>& r_ids, host_vector<soa::name_type>& r_names, 
	host_vector<soa::arity_type>& r_arities, 
	host_vector<soa::port_type>& r_p1s, 
	host_vector<soa::id_type>& r_p2s, 
	host_vector<soa::id_type>& r_p3s, 
	host_vector<soa::id_type>& r_p4s)
{
	//-------
	//options
	bool print_final_net = config_manager::get_bool("print_final");
	bool verbose = config_manager::get_bool("verbose");
	bool pause = config_manager::get_bool("pause");
	const int ruleset = config_manager::get_int("ruleset");

	//timer
	const unsigned int mywatch = StopWatch::create();
	StopWatch::get(mywatch).start();

	/* ----------------
	// interaction loop
	   ----------------  */
	//arbitrary_transformation_example();
	result_info rinfo = 
	//	test_interaction_step(in_lhs_host, in_rhs_host, verbose, ins_type, print_final_net);
		soa::interaction_loop(ruleset, l_ids, l_names, l_arities, l_p1s, l_p2s, l_p3s, l_p4s,r_ids, r_names, r_arities, r_p1s, r_p2s, r_p3s, r_p4s, verbose, print_final_net, pause);

	StopWatch::get(mywatch).stop();

	// stop and destroy timer
	float time_elapsed = StopWatch::get(mywatch).getTime() / 1000.0;
	printf ("elapsed time %.5f seconds.\n------------------\n", time_elapsed);
	StopWatch::destroy(mywatch);

	std::cout << "total loops: " << rinfo.num_loops << std::endl;
	std::cout << "total interactions: " << rinfo.total_interactions << " ( " << ( static_cast<float>(rinfo.total_interactions / time_elapsed)) << " / sec )"  << std::endl;
	const float avg_interactions = static_cast<float>(rinfo.total_interactions) / static_cast<float>(rinfo.num_loops);
	std::cout << "average interactions per loop: " << avg_interactions << std::endl;

	std::cout << "result (number of S symbols): " << rinfo.result << std::endl;
	std::cout << "\n----------------------------\n";

	//write stuff to file:
	if (config_manager::get_bool("file_output")) {
		std::ofstream outfile;
		outfile.open("out.csv", ios::app);
		outfile 
			<< time_elapsed << ";" 
			<< rinfo.num_loops << ";"
			<< rinfo.total_interactions << ";"
			<< avg_interactions << ";"
			<< (rinfo.total_interactions / time_elapsed) << ";"
			<< rinfo.com_time << ";"
			<< (rinfo.com_time / time_elapsed) << ";"
			<< rinfo.result
			<< "\n";
		outfile.close();
	}
}


void eval_all_soa(vector< host_vector<soa::id_type> >& l_ids, vector< host_vector<soa::name_type> >& l_names, 
	vector< host_vector<soa::arity_type> >& l_arities, 
	vector< host_vector<soa::port_type> >& l_p1s, 
	vector< host_vector<soa::id_type> >& l_p2s, 
	vector< host_vector<soa::id_type> >& l_p3s, 
	vector< host_vector<soa::id_type> >& l_p4s,
	vector< host_vector<soa::id_type> >& r_ids, vector< host_vector<soa::name_type> >& r_names, 
	vector< host_vector<soa::arity_type> >& r_arities, 
	vector< host_vector<soa::port_type> >& r_p1s, 
	vector< host_vector<soa::id_type> >& r_p2s, 
	vector< host_vector<soa::id_type> >& r_p3s, 
	vector< host_vector<soa::id_type> >& r_p4s)
{
	size_t num_inputs = l_ids.size();
	std::cout << "evaluating " << num_inputs << " inputs\n";
	// file i/o
	if (config_manager::get_bool("file_output")) {
		std::ofstream outfile;
		outfile.open("out.csv");
		outfile << "elapsed time;loops;interactions; avg interactions/loop; interactions/sec; com time; com time %;result \n";
		outfile.close();
	}
	for (unsigned int i=0; i< num_inputs; ++i) {
		eval_single_soa(l_ids[i], l_names[i], l_arities[i], l_p1s[i], l_p2s[i], l_p3s[i], l_p4s[i], 
					r_ids[i], r_names[i], r_arities[i], r_p1s[i], r_p2s[i], r_p3s[i], r_p4s[i]);
	}
}

int main()
{
		//load options
		config_manager::inst()->load_config("options.txt");

		//raw_agent parsing
		// ----------------
		std::string str = "input.txt";
		std::vector<parse_net::raw_agent> ra;
		std::vector< vector<parse_net::raw_agent> > ra_vector;
		std::vector<ingpu::Agent> agents;
		std::vector< vector<ingpu::Agent> > agents_vector;
		// -------------------------
		//test building of soa agents
		// --------------------------
		vector< vector<soa::id_type> > ids;
		vector< vector<soa::name_type> > names;
		vector< vector<soa::arity_type> > arities; 
		vector< vector<soa::port_type> > p1s; 
		vector< vector<soa::port_type> > p2s; 
		vector< vector<soa::port_type> > p3s; 
		vector< vector<soa::port_type> > p4s;

		parse_net::raw_agent_parser p;
		// multi parse of all inputs
		if (parse_net::parse_all(str, ra_vector))
		{
			std::cout << "-------------------------\n";
			std::cout << "Parsing succeeded\n";
			std::cout << str << " Parses OK: " << std::endl;
			int vector_size = ra_vector.size();
			//resize soa_vectors
			ids.resize(vector_size); names.resize(vector_size); arities.resize(vector_size);
			p1s.resize(vector_size); p2s.resize(vector_size); p3s.resize(vector_size); p4s.resize(vector_size); 
			//convert to proper agents now
			agents_vector.resize(vector_size);
			for (unsigned int i=0; i< ra_vector.size(); ++i) {
				agents_vector[i] = vector<ingpu::Agent>();
				build_agents(ra_vector[i], agents_vector[i]);
				//soa version
				if (config_manager::get_bool("soa") ) {
					//init soa arrays for the current input
					ids[i] = vector<soa::id_type>(); names[i] = vector<soa::name_type>(); arities[i] = vector<soa::arity_type>();
					p1s[i] = vector<soa::port_type>(); p2s[i] = vector<soa::port_type>(); p3s[i] = vector<soa::port_type>(); p4s[i] = vector<soa::port_type>(); 
					//build soa arrays for the current input
					build_agents(ra_vector[i], ids[i], names[i], arities[i], p1s[i], p2s[i], p3s[i], p4s[i]);
				}
			}
		}
		else
		{
			std::cout << "-------------------------\n";
			std::cout << "Parsing failed\n";
			std::cout << "-------------------------\n";
		}



	//splitting agents into lhs and rhs
	thrust::host_vector<Agent> lhs;
	thrust::host_vector<Agent> rhs;
	vector< thrust::host_vector<Agent> > lhs_vector;
	vector< thrust::host_vector<Agent> > rhs_vector;

	//all inputs split
	for (unsigned int input=0; input < agents_vector.size(); ++input) {
		thrust::host_vector<Agent> lhs_temp;
		thrust::host_vector<Agent> rhs_temp;
		//fill the temporary vectors
		for (unsigned int i = 0; i < agents_vector[input].size(); ++i) {
			if ((i % 2) == 0)
				lhs_temp.push_back(agents_vector[input][i]);
			else
				rhs_temp.push_back(agents_vector[input][i]);
		}
		//append temporary vectors
		lhs_vector.push_back(lhs_temp);
		rhs_vector.push_back(rhs_temp);
	}

	//--------------
	// soa splitting
	//--------------

	vector< host_vector<soa::id_type> > l_ids_vector, r_ids_vector;
	vector< host_vector<soa::name_type> > l_names_vector, r_names_vector;
	vector< host_vector<soa::arity_type> > l_arities_vector, r_arities_vector; 
	vector< host_vector<soa::port_type> > l_p1s_vector, r_p1s_vector; 
	vector< host_vector<soa::port_type> > l_p2s_vector, r_p2s_vector; 
	vector< host_vector<soa::port_type> > l_p3s_vector, r_p3s_vector; 
	vector< host_vector<soa::port_type> > l_p4s_vector, r_p4s_vector;
	
	if (config_manager::get_bool("soa") )
	{
	//outer loop
	for (unsigned int input = 0; input < ids.size(); ++input) {
		// temporary arrays
		host_vector<soa::id_type> l_ids, r_ids;
		host_vector<soa::name_type> l_names, r_names;
		host_vector<soa::arity_type> l_arities, r_arities; 
		host_vector<soa::port_type> l_p1s, r_p1s; 
		host_vector<soa::port_type> l_p2s, r_p2s; 
		host_vector<soa::port_type> l_p3s, r_p3s; 
		host_vector<soa::port_type> l_p4s, r_p4s;
	
		//split arrays for the current input
		for (unsigned int i = 0; i < ids[input].size(); ++i) {
			if ((i % 2) == 0) {
				l_ids.push_back(ids[input][i]);
				l_names.push_back(names[input][i]);
				l_arities.push_back(arities[input][i]);
				l_p1s.push_back(p1s[input][i]);
				l_p2s.push_back(p2s[input][i]);
				l_p3s.push_back(p3s[input][i]);
				l_p4s.push_back(p4s[input][i]);
			}
			else {
				r_ids.push_back(ids[input][i]);
				r_names.push_back(names[input][i]);
				r_arities.push_back(arities[input][i]);
				r_p1s.push_back(p1s[input][i]);
				r_p2s.push_back(p2s[input][i]);
				r_p3s.push_back(p3s[input][i]);
				r_p4s.push_back(p4s[input][i]);
			}
		}

		// add the arrays to the input vector
		// lhs
		l_ids_vector.push_back(l_ids);
		l_names_vector.push_back(l_names);
		l_arities_vector.push_back(l_arities);
		l_p1s_vector.push_back(l_p1s);
		l_p2s_vector.push_back(l_p2s);
		l_p3s_vector.push_back(l_p3s);
		l_p4s_vector.push_back(l_p4s);
		// rhs
		r_ids_vector.push_back(r_ids);
		r_names_vector.push_back(r_names);
		r_arities_vector.push_back(r_arities);
		r_p1s_vector.push_back(r_p1s);
		r_p2s_vector.push_back(r_p2s);
		r_p3s_vector.push_back(r_p3s);
		r_p4s_vector.push_back(r_p4s);
	}
	}

	//evaluation
	//----------
	//soa
	if ( config_manager::get_bool("soa")) {
		
		std::cout << "data mode: soa\n";
		if (config_manager::get_bool("eval_all") ) {
			eval_all_soa(l_ids_vector, l_names_vector, l_arities_vector, l_p1s_vector, l_p2s_vector, l_p3s_vector, l_p4s_vector, 
						r_ids_vector, r_names_vector, r_arities_vector, r_p1s_vector, r_p2s_vector, r_p3s_vector, r_p4s_vector);
		}
		else {
			eval_single_soa(l_ids_vector[0], l_names_vector[0], l_arities_vector[0], l_p1s_vector[0], l_p2s_vector[0], l_p3s_vector[0], l_p4s_vector[0], 
						r_ids_vector[0], r_names_vector[0], r_arities_vector[0], r_p1s_vector[0], r_p2s_vector[0], r_p3s_vector[0], r_p4s_vector[0]);
		}
	}
	//regular evaluation
	else {
		if (config_manager::get_bool("partition")) {
			//std::cout << "data mode: partition\n";
		}
		else {
			std::cout << "data mode: regular\n";
		}
		if (config_manager::get_bool("eval_all") ) {
			eval_all(lhs_vector, rhs_vector);
		}
		else {
			eval_single(lhs_vector[0], rhs_vector[0]);
		}
	}
	//// ------
	////options
	//bool print_final_net = config_manager::get_bool("print_final");
	//bool verbose = config_manager::get_bool("verbose");
	//int ins_type = config_manager::get_int("ins");

	////timer
	//const unsigned int mywatch = StopWatch::create();
	//StopWatch::get(mywatch).start();

	///* ----------------
	//// interaction loop
	//   ----------------  */
	////arbitrary_transformation_example();
	//result_info rinfo = 
	//	test_interaction_step(lhs, rhs, verbose, ins_type, print_final_net);
	////	soa::interaction_loop( l_ids, l_names, l_arities, l_p1s, l_p2s, l_p3s, l_p4s,r_ids, r_names, r_arities, r_p1s, r_p2s, r_p3s, r_p4s, verbose, print_final_net);

	/* ----------------
	// interaction loop
	   ----------------  */
	//arbitrary_transformation_example();
	//test_interaction_step(lhs, rhs, verbose, ins_type, print_final_net);
	//soa::interaction_loop( l_ids, l_names, l_arities, l_p1s, l_p2s, l_p3s, l_p4s,r_ids, r_names, r_arities, r_p1s, r_p2s, r_p3s, r_p4s, verbose, print_final_net, pause);
	//StopWatch::get(mywatch).stop();

	//// stop and destroy timer
	//printf ("elapsed time %.5f seconds.\n------------------\n", StopWatch::get(mywatch).getTime() / 1000.0 );
	//StopWatch::destroy(mywatch);

	//std::cout << "total loops: " << rinfo.num_loops << std::endl;
	//std::cout << "total interactions: " << rinfo.total_interactions << std::endl;
	//const float avg_interactions = static_cast<float>(rinfo.total_interactions) / static_cast<float>(rinfo.num_loops);
	//std::cout << "average interactions per loop: " << avg_interactions << std::endl;

	//std::cout << "result (number of S symbols): " << rinfo.result << std::endl;

	std::cout << "press any key to continue...\n";
	getch();

	return 0;
}