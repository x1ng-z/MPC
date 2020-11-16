
import gc
import os
import sys
import traceback



this = os.path.abspath(os.path.dirname(__file__))
module = os.path.split(this)[0]
#print('sys.path.append("%s")' % module)
sys.path.append(module)
for i, val in enumerate(sys.path):
    pass
    #print("[%s] %s" % (i + 1, val))


def delMEI():
    this = os.path.abspath(os.path.dirname(__file__))
    module = os.path.split(this)[0]
    # print('sys.path.append("%s")' % module)
    sys.path.append(module)
    for i, val in enumerate(sys.path):
        pass
        # print("[%s] %s" % (i + 1, val))

    for index, path in enumerate(sys.path):
        basename = os.path.basename(path)
        if not basename.startswith("_MEI"):
            continue

        drive = os.path.splitdrive(path)[0]
        if "" == drive:
            path = os.getcwd() + "\\" + path
            path = path.replace("\\\\", "\\")

        if os.path.isdir(path):
            try:
                #print("remove", path)
                os.remove(path)
            except:
                pass
            finally:
                break


if __name__ == '__main__':
    try:
        print("Start to xx..")
        # do something
        print("End of the xx")
    except:
        traceback.print_exc()
    finally:
        gc.collect()
        #delMEI()