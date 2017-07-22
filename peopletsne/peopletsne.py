# coding: utf-8

from __future__ import division, print_function
from collections import namedtuple,defaultdict
from bokeh.plotting import figure, output_notebook, output_file, show
from ipywidgets import fixed, interact, interact_manual, interactive
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from face import face
import face_recognition_models
import pickle
import cv2
import ipywidgets as widgets
import glob
import os
import numpy as np
import math
import PIL
from bokeh.plotting import *
from bokeh.models.tools import *
from bokeh.models import CustomJS

def retImg(in_img):
    s = in_img.shape
    x = s[0]
    y = s[1]
    img = np.empty((x, y), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((x, y, 4))
    for i in range(x):
        for j in range(y):
            val = in_img[x-i-1,j]
            view[i, j, 0] = val[2]
            view[i, j, 1] = val[1]
            view[i, j, 2] = val[0]
            view[i, j, 3] = 255
    return img


#point this to a directory of directories containing example cropped images of ppl of interest
def make_tasking(lookupdir,outfilename):
    tasking = defaultdict(dict)
    for tdir in glob.glob(lookupdir):
        encodings = []
        pics = []
        print(tdir)
        for qfile in glob.glob(os.path.join(tdir,'*')):
            print(qfile)
            face_image = cv2.imread(qfile)
            locs = face.face_locations(face_image)
            enc = face.face_encodings(face_image, None)
            if enc and len(enc) >=1:
                print(qfile)
                top, right, bottom, left = locs[0]
                encodings.append(face.face_encodings(face_image, None)[0])
                cv2.rectangle(face_image, (left, top),
                              (right, bottom), (0, 255, 0), 2)
                pics.append(face_image[top:bottom, left:right])
        key = tdir.split('/')[-1]
        tasking[key]['face_vec']= encodings
        tasking[key]['pic'] = pics
    pickle.dump(tasking,open(outfilename,'wb'))


#remember to mount /in
#then mkdir /tmp/in
#run webserver homed in exporteddir/tmp
#then you get mouseover pics..

#plot dots where faces are, black dots for video data, colored dots for people of interest
def tsne_faces(prep,X,COLOR,lr,d,thumbs):
    model = TSNE(n_components=2, random_state=234)
    model.perplexity = prep
    model.learning_rate = lr
    d['prep']=prep
    d['lr']=lr
   
    np.set_printoptions(suppress=False)
    out= model.fit_transform(X) 
    # plot the result
    vis_x = out[:, 0]
    vis_y = out[:, 1]
    
    hover = HoverTool( tooltips="""
    <div>
        <div>
            <img
                src="@pics" height="42" width="42"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="0"
            ></img>
            <span style="font-size: 15px; color: #966;">$index</span>
        </div>
        <div>
            <span style="font-size: 10px; color: #696;">(@xvals, @yvals)</span>
        </div>
    </div>
    """
    )
    
    TOOLS="pan,wheel_zoom,reset,lasso_select"
    p = figure(title = "faceLocations",tools=TOOLS,output_backend="webgl")#,figsize=(15, 15))
    
    p.add_tools(hover)
    
    source = ColumnDataSource(
    data=dict(
        xvals=list(vis_x),
        yvals=list(vis_y),
        fill_color=list(COLOR),
        pics = list(thumbs)
        )
    )
    
    p.scatter("xvals","yvals",color='fill_color',source=source,fill_alpha=0.2,size=5)
    
    source.callback = CustomJS(args=dict(p=p), code="""
        var inds = cb_obj.get('selected')['1d'].indices;
        var d1 = cb_obj.get('data');
        var dat_len = inds.length;
        //console.log(d1)
        
        var js_tsne_x = new Array();
        var js_tsne_y = new Array();
        
        //console.log(inds);
        
        //console.log('starting');
        //console.log('length:' + dat_len)
        for (i=0; i< dat_len; i++){
            js_tsne_x.push(d1.xvals[inds[i]]);
            js_tsne_y.push(d1.yvals[inds[i]]);
        }
        
        //console.log(js_tsne_x);
        //console.log(js_tsne_y);
        
        var kernel = IPython.notebook.kernel;
        IPython.notebook.kernel.execute("inds = " + inds);
        IPython.notebook.kernel.execute("tsne_x = "+ js_tsne_x);
        IPython.notebook.kernel.execute("tsne_y = "+ js_tsne_y);
        
        """
    )
    
    show(p)
    return (prep,lr)


#plot pictures blue frame is video data, green frame is where a query is

def tsne_html(datastore,d,spread=50,tsne=True):
    X,COLOR,IMG,dw,dh,xs,ys = datastore
    model = TSNE(n_components=2, random_state=234)
    model.perplexity=d['prep']
    model.learning_rate = d['lr']
    if tsne:
        np.set_printoptions(suppress=False)
        out= model.fit_transform(X) 
        # plot the result
        vis_x = out[:, 0]*spread
        vis_y = out[:, 1]*spread
    else:
        vis_x = np.array(xs)*spread
        vis_y = np.array(ys)*spread
        
    p = figure(plot_width=800, plot_height=800, x_range=(-80, 80), y_range=(-80,80),output_backend="webgl")

    p.image_rgba(image=IMG , x=vis_x , y=vis_y , dw=dw , dh=dh )

    show(p)    
    #output_file("faces.html")
    

def makestats(items):
    labels=[]
    detections=0
    for item in items:
        detections += item[1]
        labels.append(item[0])
    return 'labels:{0} detections:{1}'.format(labels,detections)

def showResults(threshold,key, tasking ,people):
    found = list()
    uniq = set()
    #for enc in encodings:
    #print(type(all_enc),type(sensativity))
    for enc in tasking[key]['face_vec']:
        for candidate in people:
            c_enc = people[candidate]['face_vec']
            c_pic = people[candidate]['pic']
            c_tim = people[candidate]['times']
            dist = face.face_distance([enc],c_enc)
            if dist < threshold:
                #print ('candidate:{0} detections:{1} dist:{2:0.3}'.format(candidate,len(c_tim),dist[0]))
                RGB_img = cv2.cvtColor(c_pic, cv2.COLOR_BGR2RGB)
                if candidate not in uniq:
                    uniq.add(candidate)
                    found.append((candidate,len(c_tim),dist[0],RGB_img))
    
    side = math.ceil(math.sqrt(len(found)))
    plt.figure(figsize=(20, 20))
    sorted_by_distance = sorted(found, key=lambda tup: tup[2])
    
    stats = makestats(sorted_by_distance)

    for idx, f in enumerate(sorted_by_distance):
        plt.subplot(side,side,idx+1)
        plt.title("label:{0} dist:{1:02.3} det:{2:03}".format(f[0],f[2],f[1]))
        plt.axis('off')
        plt.imshow(PIL.Image.fromarray(f[3]))
    print(stats)
    plt.show()


def reduce_datastore(datastore, inds,xs,ys,recenter=True):

    vec = []
    color = []
    pic = []
    dw = []
    dh = []
    plot_x = []
    plot_y = []
    insert =0
    print("selected:",len(inds)," items")
    for idx,x in enumerate(inds):
        insert += 1
        vec.append(datastore[0][x])
        color.append(datastore[1][x])
        pic.append(datastore[2][x])
        dw.append(datastore[3][x])
        dh.append(datastore[4][x])
        plot_x.append(xs[idx])
        plot_y.append(ys[idx])
   
    if recenter:
        #center the images in the next plot    
        plot_x = np.array(plot_x) - np.average(np.array(plot_x))
        plot_y = np.array(plot_y) - np.average(np.array(plot_y))

    print ("inserted:",insert," items")
    return (vec,color,pic,dw,dh,plot_x,plot_y)

def relabel_found(datastore,inds,name2inds,name,COLOR='red'):
    indxxs = set()
    for idx,x in enumerate(inds):
        datastore[1][x] = COLOR
        indxxs.add(x)
    name2inds[name].update(indxxs)


#compile the pickles into something to hand to tsne for viz
def make_datastore(people,tasking=None):

    X = list()
    COLOR = list()
    IMG = list()
    dw = list()
    dh = list()
    f = list()

    colors= ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 
             'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 
             'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 
             'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 
             'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 
             'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 
             'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 
             'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 
             'hotpink', 'indianred ', 'indigo ', 'ivory', 'khaki', 'lavender', 'lavenderblush', 
             'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 
             'lightgray', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 
             'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 
             'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 
             'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 
             'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 
             'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 
             'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 
             'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 
             'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 
             'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 
             'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
    
    
    for p in people:
        if not os.path.isdir("/in/tmp"): os.mkdir("/in/tmp")
        outfile_name = '/in/tmp/'+p+'.png'
        dat = people[p]['face_vec']
        COLOR.append('black')
        X.append(dat)
        pix = retImg(people[p]['pic'])
        IMG.append(pix)
        #RGB_img = cv2.cvtColor(pix, cv2.COLOR_BGR2RGB)
        #Image.fromarray(pix).save(outfile_name)
        RGB_img = Image.fromarray( cv2.cvtColor(people[p]['pic'], cv2.COLOR_BGR2RGB))
        RGB_img.save(open(outfile_name,'wb'))
        dw.append(10)
        dh.append(10)
        #f.append('file:///Users/dgrossman/Downloads/tmp/'+p+'.png')
        f.append('http://localhost:8080/'+p+'.png')
    
    idx=0
    if tasking is not None:
        for p in tasking:
           

            for x in tasking[p]['face_vec']:
                COLOR.append(colors[idx % len(colors)])
                X.append(x)

            for x in tasking[p]['pic']:
                IMG.append(retImg(x))
                dw.append(10)
                dh.append(10)
            idx +=1
    return(X,COLOR,IMG,dw,dh,f)
