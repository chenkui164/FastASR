#include <Model.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class PyModel {
  private:
    Model *mm;

  public:
    PyModel(const char *path, int mode)
    {
        mm = create_model(path, mode);
    };

    void reset()
    {
        mm->reset();
    };

    std::string forward_chunk(py::array_t<float> &din, int flag)
    {
        py::buffer_info audio_buff = din.request();
        return mm->forward_chunk((float *)audio_buff.ptr, audio_buff.size, flag);
    };

    std::string forward(py::array_t<float> &din)
    {

        py::buffer_info audio_buff = din.request();
        return mm->forward((float *)audio_buff.ptr, audio_buff.size, 2);
    };

    std::string rescoring()
    {
        return mm->rescoring();
    };
};

PYBIND11_MODULE(_fastasr, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring
                                         //

    py::class_<PyModel>(m, "Model")
        .def(py::init<const char *, int>())
        .def("reset", &PyModel::reset)
        .def("forward_chunk", &PyModel::forward_chunk)
        .def("forward", &PyModel::forward)
        .def("rescoring", &PyModel::rescoring);
}
