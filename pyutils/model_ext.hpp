/****************************************************************************************
 * Extension C++ module for 1D firm simulations. 
 *
 * Author : Eddie Lee, edlee@csh.ac.at
 ****************************************************************************************/
#include <stdio.h>
#include <iostream>
#include <ostream>
#include <vector>
#include <assert.h>
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numpy.hpp>

#define BOOST_TEST_DYN_LINK

using namespace std;
namespace py = boost::python;
namespace np = boost::python::numpy;



// create numpy array
np::ndarray vec2ndarray(const vector<int> &x) {
    //The commented code doesn't work! I don't know why because it comes from a Boost tutorial.
    //int sample_size = x.size();
    //py::tuple shape = py::make_tuple(sample_size);
    //py::tuple stride = py::make_tuple(sizeof(vector<int>::value_type));
    //np::dtype dt = np::dtype::get_builtin<int>();
    //np::ndarray output = np::from_data(&x, dt, shape, stride, py::object());

    //for (int i=0; i<4; i++) {
    //    cout << py::extract<int>(output[i]) << ", ";
    //};
    //cout << endl;

    //return output;

    Py_intptr_t shape[1] = { x.size() };
    np::ndarray result = np::zeros(1, shape, np::dtype::get_builtin<int>());
    copy(x.begin(), x.end(), reinterpret_cast<int*>(result.get_data()));
    return result;
};



class TopicLattice {
    public: 

    int left = 0;
    int right = 0;
    vector<int> occupancy = vector<int>(1, 0);
    vector<int> d_occupancy = vector<int>(1, 0);

    /******************
     * Info functions *
     ******************/
    int get_occ(int i) {
        if ((i>=left) & (i<=right)) {
            return occupancy[i-left];
        };
        return 0;
    };
    
    // Takes vector of indices and returns occupancy at specified sites.
    np::ndarray get_occ_vec(np::ndarray &i) {
        np::ndarray occ = np::empty(py::make_tuple(i.shape(0)), i.get_dtype());
        int ix;

        for (int counter=0; counter<i.shape(0); counter++) {
            if ((i[counter]>=left) & (i[counter]<=right)) {
                ix = py::extract<int>(i[counter]);
                occ[counter] = occupancy[ix-left];
            };
        };
        return occ;
    };

    int len() {
        return occupancy.size();
    };

    np::ndarray view() {
        return vec2ndarray(occupancy);
        //np::ndarray output_copy =  vec2ndarray(occupancy).copy();
        //return output_copy;
    };

    /* ============= *
     * Mod functions *
     * ============= */
    void shrink_left(int n=1) {
        if ((n>=1) &  (left+n<=right)) {
            left += n;
            occupancy.erase(occupancy.begin(), occupancy.begin()+n);
            d_occupancy.erase(d_occupancy.begin(), d_occupancy.begin()+n);
        };
    };

    //Parameters
    //----------
    //n : int, 1
    //    Number of spots by which to extend lattice.
    void extend_left(int n=1) {
        left -= n;
        occupancy.insert(occupancy.begin(), n, 0);
        d_occupancy.insert(d_occupancy.begin(), n, 0);
    };

    //Parameters
    //----------
    //n : int, 1
    //    Number of spots by which to extend lattice.
    void extend_right(int n=1) {
        occupancy.insert(occupancy.end(), n, 0);
        d_occupancy.insert(d_occupancy.end(), n, 0);
        right += n;
    };

    //Remove one occupant from lattice site i.
    // 
    //Parameters
    //----------
    //i : int
    //    This is the coordinate and not the number of bins from the leftmost spot.
    //d : int, 1
    void remove(int i, int d=1) {
        if ((i<left) | (i>right)) {
             PyErr_SetString(PyExc_ValueError, "incrementing outside of lattice range");
           py::throw_error_already_set();
        };
        occupancy[i-left] -= d;
    };

    //Add one occupant to lattice site i.
    //
    //Parameters
    //----------
    //i : int
    //    This is the coordinate and not the number of bins from the leftmost spot.
    //d : int, 1       
    void add(int i, int d=1) {
        if ((i<left) | (i>right)) {
             PyErr_SetString(PyExc_ValueError, "incrementing outside of lattice range");
           py::throw_error_already_set();
        };
        occupancy[i-left] += d;
    };
       
    //Remove one occupant from lattice site i for delayed occupancy.
    // 
    //Parameters
    //----------
    //i : int
    //    This is the coordinate and not the number of bins from the leftmost spot.
    //d : int, 1
    void d_remove(int i, int d=1) {
        if ((i<left) | (i>right)) {
             PyErr_SetString(PyExc_ValueError, "incrementing outside of lattice range");
           py::throw_error_already_set();
        };
        d_occupancy[i-left] -= d;
    };

    //Add one occupant to lattice site i for delayed occupancy.
    //
    //Parameters
    //----------
    //i : int
    //    This is the coordinate and not the number of bins from the leftmost spot.
    //d : int, 1       
    void d_add(int i, int d=1) {
       if ((i<left) | (i>right)) {
            PyErr_SetString(PyExc_ValueError, "incrementing outside of lattice range");
          py::throw_error_already_set();
       };
       d_occupancy[i-left] += d;
    };
 
    //Push stored changes to occupancy and reset self.d_occupancy.
    void push() {
        for (int i; i<occupancy.size(); i++) {
            occupancy[i] += d_occupancy[i];
            d_occupancy[i] = 0;
        };
    };
    
    //Reset occupancy values to 0.
    void clear() {
        for (vector<int>::iterator it=occupancy.begin(); it!=occupancy.end(); it++) {
            *it = 0;
        };
        return;
    };

    //Reset d_occupancy values to 0.
    void d_clear() {
        for (vector<int>::iterator it=d_occupancy.begin(); it!=d_occupancy.end(); it++) {
            *it = 0;
        };
        return;
    };
        
    //Copy the topic lattice.
    TopicLattice copy() {
        TopicLattice copy;
        copy.left = left;
        copy.right = right;
        copy.occupancy = vector<int>(occupancy.size(), 0);
        for (int i; i<occupancy.size(); i++) {
            copy.occupancy[i] = occupancy[i];
        };
        return copy;
    };
};
//end TopicLattice


//pickle suite necessary for cpickle interface
struct TopicLatticePickleSuite : py::pickle_suite {
    static py::tuple getstate(py::object lattice_obj)
    {
        TopicLattice& lattice = py::extract<TopicLattice&>(lattice_obj)();
        return py::make_tuple(lattice.left, lattice.right, lattice.occupancy, lattice.d_occupancy);
    }

    static void setstate(py::object lattice_obj, py::tuple state) {
        if (py::len(state) != 4) {
            PyErr_SetObject(PyExc_ValueError,
                            ("expected 4-item tuple in call to __setstate__; got %s"
                             % state).ptr()
                            );
          py::throw_error_already_set();
        }

        TopicLattice& lattice = py::extract<TopicLattice&>(lattice_obj)();
        lattice.left = py::extract<int>(state[0])();
        lattice.right = py::extract<int>(state[1])();
        lattice.occupancy = py::extract<vector<int>&>(state[2])();
        lattice.d_occupancy = py::extract<vector<int>&>(state[3])();

        // restore the object's __dict__
        //py::dict d = py::extract<py::dict>(firm_obj.attr("__dict__"))();
        //d.update(state[0]);
    }

    static bool getstate_manages_dict() { return false; }
};




class LiteFirm {
    public:
    //Parameters
    //----------
    //sites : tuple
    //    Left and right boundaries of firm. If a int, then it assumes that firm is
    //    localized to a single site.
    //innov : float
    //    Innovation parameter. This determines the probability with which the firm
    //    will choose to expand into a new region or go into an already explored region.
    //    This is much more interesting in topic space beyond one dimension.
    //connection_cost : float, 0.
    //wealth : float, 1.
    //age : int
    //id : str
    py::object sites;
    double innov;
    double wealth;
    double connection_cost;
    int age;
    string id;
    py::dict dict;

    LiteFirm() {};

    LiteFirm(py::object new_sites,
             double new_innov,
             double new_wealth,
             double new_connection_cost,
             int new_age,
             string new_id) {
        sites = new_sites;
        innov = new_innov;
        wealth = new_wealth;
        connection_cost = new_connection_cost;
        age = new_age;
        id = string(new_id);
    };
};//end LiteFirm


//pickle suite necessary for cpickle interface
struct LiteFirmPickleSuite : py::pickle_suite {
    static py::tuple getinitargs(const LiteFirm& firm)
    {
        return py::make_tuple(py::make_tuple(0,0), 0., 0., 0., 0, "must init");
    }

    static py::tuple getstate(py::object firm_obj)
    {
        LiteFirm& firm = py::extract<LiteFirm&>(firm_obj)();
        return py::make_tuple(firm.sites,
                              firm.innov,
                              firm.wealth,
                              firm.connection_cost,
                              firm.age,
                              firm.id);
    }

    static void setstate(py::object firm_obj, py::tuple state) {
        if (py::len(state) != 6) {
            PyErr_SetObject(PyExc_ValueError,
                            ("expected 6-item tuple in call to __setstate__; got %s"
                             % state).ptr()
              );
          py::throw_error_already_set();
        }

        LiteFirm& firm = py::extract<LiteFirm&>(firm_obj)();
        firm.sites = state[0];
        firm.innov = py::extract<double>(state[1])();
        firm.wealth = py::extract<double>(state[2])();
        firm.connection_cost = py::extract<double>(state[3])();
        firm.age = py::extract<int>(state[4])();
        firm.id = py::extract<string>(state[5])();

        // restore the object's __dict__
        //py::dict d = py::extract<py::dict>(firm_obj.attr("__dict__"))();
        //d.update(state[0]);
    }

    static bool getstate_manages_dict() { return false; }
};



/*********************************
 * Useful functions for LiteFirm *
 *********************************/
//this doesn't work...all I wanted to do was to speed up a for loop in Python here
py::list snap_firms(py::list& firms) {
    py::object model = py::import("pyutils.model");

    py::list snapshot;
    py::object Firm = model.attr("Firm");
    py::object f = Firm(py::make_tuple(1,2), .3);

    for (int i=0; i<py::len(firms); i++) {
        f = py::extract<py::object>(firms[i])();
        //py::call_method<LiteFirm&>(f, "copy");
        //f = py::extract<py::object&>(firms[i])();
        //snapshot.append(py::extract<LiteFirm&>(PyObject_CallMethod(f, "copy", "()"))());
        i++;
    };

    return snapshot;
};



/*********************************
 * Wrappers for Python interface *
 ********************************/
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(add_over, add, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(remove_over, remove, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(d_add_over, d_add, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(d_add_remove, d_remove, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(shrink_left_over, shrink_left, 0, 1);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(extend_left_over, extend_left, 0, 1);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(extend_right_over, extend_right, 0, 1);

BOOST_PYTHON_MODULE(model_ext) {
    using namespace boost::python;
    Py_Initialize();
    np::initialize();

    class_<TopicLattice>("TopicLattice", init<>())
        .def_readonly("left", &TopicLattice::left)
        .def_readonly("right", &TopicLattice::right)
        .def("get_occ", &TopicLattice::get_occ)
        .def("get_occ", &TopicLattice::get_occ_vec)
        .def("len", &TopicLattice::len)
        .def("view", &TopicLattice::view)
        .def("shrink_left", &TopicLattice::shrink_left, shrink_left_over())
        .def("extend_left", &TopicLattice::extend_left, extend_left_over())
        .def("extend_right", &TopicLattice::extend_right, extend_right_over())
        .def("add", &TopicLattice::add, add_over())
        .def("remove", &TopicLattice::remove, remove_over())
        .def("d_add", &TopicLattice::d_add, d_add_over())
        .def("d_remove", &TopicLattice::d_remove)
        .def("push", &TopicLattice::push)
        .def("clear", &TopicLattice::clear)
        .def("d_clear", &TopicLattice::d_clear)
        .def("copy", &TopicLattice::copy)
    ;

    class_<LiteFirm>("LiteFirm", init<py::object, double, double, double, int, string>())
        .def_readonly("sites", &LiteFirm::sites)
        .def_readonly("innov", &LiteFirm::innov)
        .def_readonly("wealth", &LiteFirm::wealth)
        .def_readonly("connection_cost", &LiteFirm::connection_cost)
        .def_readonly("age", &LiteFirm::age)
        .def_readonly("id", &LiteFirm::id)
        .def_pickle(LiteFirmPickleSuite())
    ;
}
