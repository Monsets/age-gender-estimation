all:
	cd rcnn_ag/cython/; python3 setup.py build_ext --inplace; rm -rf build; cd ../../
	#cd rcnn/pycocotools/; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd rcnn_ag/cython/; rm *.so *.c *.cpp; cd ../../
	#cd rcnn/pycocotools/; rm *.so; cd ../../
