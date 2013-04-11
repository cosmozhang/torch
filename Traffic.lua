----------------------------------------------------------------------
-- E. Culurciello Fall 2012
-- Run k-means on Berkeley image and generate layers filters
-- simulate the Online Learner (OL) network as a robotic vision template
----------------------------------------------------------------------

import 'torch'
require 'csv'
require 'image'
require 'unsup'
require 'nnx'
require 'eex'
require 'nn'
--require 'MulAnySize'

--dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Get k-means templates on directory of images')
cmd:text()
cmd:text('Options')
cmd:option('-datafile', '../GTSRB/Final_Training/Images/', 'Dataset path')
--cmd:option('-datafile', '../datasets/faces_cut_yuv_32x32/face/', 'Dataset path')
--cmd:option{'-d', '--datafile', action='store', dest='datafile', default='../datasets/faces_cut_yuv_32x32/', help='path to MNIST root dir'}
cmd:option('-testfile', '../GTSRB/Final_Test/Images/', 'Testset path')
cmd:option('--www', 'http://data.neuflow.org/data/faces_cut_yuv_32x32.tar.gz', 'Dataset URL')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 5, 'size of each input patches') -- OL: 7x7
cmd:option('-nkernels1', 16, 'number of kernels 1st layer') -- OL: 16
cmd:option('-nkernels2', 128, 'number of kernels 2nd layer') -- OL: 128
cmd:option('-nkernels3', 128, 'number of kernels 3rd layer') -- OL: 128
cmd:option('-niter1', 15, 'nb of k-means iterations')
cmd:option('-niter2', 15, 'nb of k-means iterations')
cmd:option('-niter3', 15, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 10*1000, 'nb of random training samples')
cmd:option('-initstd1', 0.1, 'standard deviation to generate random initial templates')
cmd:option('-initstd2', 0.02, 'standard deviation to generate random initial templates')
cmd:option('-initstd3', 0.01, 'standard deviation to generate random initial templates')
cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-loss', 'mse', 'type of loss function to minimize: nll | mse | margin')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-plot', true, 'live plot')
cmd:option('-visualize', 1, 'visualize the datasets')
cmd:option('-patches', 'all', 'nb of patches to use')
cmd:text()
opt = cmd:parse(arg or {}) -- pass parameters to rest of file:

function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

--if not qt then
--   opt.visualize = false
--end

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

is = opt.inputsize
nk1 = opt.nkernels1
nk2 = opt.nkernels2
nk3 = opt.nkernels3

normkernel = image.gaussian1D(5)

----------------------------------------------------------------------
-- loading and processing dataset:
--dofile '1_data.lua'

if not sys.dirp(opt.datafile) then
--[[
   local path = sys.dirname(opt.datafile)
   local tar = sys.basename(opt.www)
   os.execute('mkdir -p ' .. path .. '; '..
              'cd ' .. path .. '; '..
              'wget ' .. opt.www .. '; '..
              'tar xvf ' .. tar)
]]
   print('Error! Path wrong')
end
--filename=sys.dirname(opt.datafile)
--filename = paths.basename(opt.datafile)
--print('filename', filename)
--if not sys.dirp(opt.datafile) then
--   os.execute('wget ' .. opt.datafile .. '; '.. 'tar xvf ' .. filename)
--end
--dataset = getdata(filename, opt.inputsize)
--[[
trsize = 256--dataset:size()
tssize = 128

dataset = {
   data = torch.Tensor(trsize+tssize, 3, 32, 32),
   labels = torch.Tensor(trsize+tssize, 2),
   size = function() return trsize end
}
]]
trainDataload = nn.DataList()

for i=0, 1 do
   if i <10 then
      fname = '0000'..i
      cl='0'..i
   else
      fname = '000'..i
      cl=''..i
   end

data = nn.DataSet{dataSetFolder=opt.datafile..fname, 
		     cacheFile=opt.datafile..fname,
		     nbSamplesRequired=opt.patches,
		     channels=1}
   --print(type(cl))

   data:shuffle()
   trainDataload:appendDataSet(data, cl)
end

trainDataload:shuffle()
trsize=trainDataload.nbSamples
print(trsize)
nclass=trainDataload.nbClass
print(nclass)
imsize=32
shownb=16

--trainData = nn.DataList()
--trainData:appendDataSet(data,'00')
--print(trainDataload.nbSample)
win4 = image.display{image={trainDataload[200][1]}, zoom=4, win=win4, legend='orig'} 

--[[ display
   if opt.visualize then
   data00:display(100,'trainData')
   --testData:display(100,'testData')
   end
]]



--sleep(10)

--image.display{image=dataset.data[6][1], zoom=2, win=win10}
------------------------------------------------------------------------------------
----------------load signs----------------------------------------------------------
------------------------------------------------------------------------------------

trainData = {
   data = torch.Tensor(trsize, 3, imsize, imsize),
   labels = torch.Tensor(trsize, nclass),
   size = function() return trsize end
}



for t = 1, trsize do
   trainData.data[t][1] = image.scale(trainDataload[t][1], imsize, imsize)
   trainData.data[t][2] = trainData.data[t][1]
   trainData.data[t][3] = trainData.data[t][1]
   trainData.labels[t] =  trainDataload[t][2]
   xlua.progress(t, trainData:size())
end
f256S = trainData.data[{{1,shownb}}]
image.display{image=f256S, nrow=math.sqrt(shownb), nrow=(shownb), padding=2, zoom=1, legend='Input images'}
print(trainData.labels[{{1,shownb}}])
------------------------------------------------------------

testinfo=csv.load('../GTSRB/Final_Test/GT-final_test.csv', ';', 'tidy')

tssize=0
for i =1, 100 do--#testinfo.Filename do
   if testinfo.ClassId[i]+0 < nclass then
      tssize=tssize+1
   end
end
print(tssize)

--[[
testDataload = nn.DataSet{dataSetFolder=opt.testfile, 
			  cacheFile=opt.testfile,
			  nbSamplesRequired=100,--opt.patches,
			  channels=1}
]]

testData = {
   data = torch.Tensor(tssize, 3, imsize, imsize),
   labels = torch.Tensor(tssize, nclass),
   size = function() return tssize end
}

x=0
for t = 1, 100 do--#testinfo.Filename do
   if testinfo.ClassId[t]+0 < nclass then
      x=x+1
      testData.data[x][1] = image.scale(image.load(opt.testfile..testinfo.Filename[t])[1], imsize, imsize)
      testData.data[x][2] = testData.data[x][1]
      testData.data[x][3] = testData.data[x][1]
      testData.labels[x][{{1,nclass}}] = -1
      --print('tl', testData.labels[x][1])
      --testData.labels[x][1]=testinfo.ClassId[t]+0
      testData.labels[x][testinfo.ClassId[t]+0]=testData.labels[x][testinfo.ClassId[t]+0]*(-1)
      xlua.progress(t, testData:size())
   end
end
f256S = testData.data[{{1,4}}]
image.display{image=f256S, nrow=2, nrow=2, padding=2, zoom=4, legend='Test images'}
print(testData.labels[{{1,4}}])

-- verify dataset statistics:
trainMean = trainData.data:mean()
trainStd = trainData.data:std()
print('1st layer training data mean: ' .. trainMean)
print('1st layer training data standard deviation: ' .. trainStd)

--print('size', dataset.data:size(1))
----------------------------------------------------------------------
print '==> generating 1st stage filters:'
print '==> extracting patches' -- only extract on Y channel (or R if RGB) -- all ok
data1 = torch.Tensor(opt.nsamples,is*is)
for i = 1,opt.nsamples do
   idx = math.random(1,trainData.data:size(1))
   img = trainData.data[idx][1]
   x = math.random(1,trainData.data[1][1]:size(1)-is+1)
   y = math.random(1,trainData.data[1][1]:size(2)-is+1)
   randompatch = img[{{y,y+is-1},{x,x+is-1}}]:clone()
   -- normalize patches to 0 mean and 1 std:
   randompatch:add(-randompatch:mean())
   --randompatch:div(randompatch:std())
   data1[i] = randompatch
end

-- show a few patches:
f256S = data1[{{1,shownb}}]:reshape(shownb,is,is)
image.display{image=f256S, nrow=math.sqrt(shownb), nrow=math.sqrt(shownb), padding=2, zoom=2, legend='Patches for 1st layer learning'}

print '==> running k-means'
 function cb (kernels1)
    if opt.visualize then
       --win1 = image.display{image=kernels1:reshape(nk1,is,is), padding=2, symmetric=true, zoom=2, win=win1, nrow=math.floor(math.sqrt(nk1)), legend='1st layer filters'}
    end
end                    
kernels1 = unsup.kmeans(data1, nk1, opt.initstd1, opt.niter1, opt.batchsize, cb, true)

-- clear nan kernels if kmeans initstd is not right!
for i=1,nk1 do   
   if torch.sum(kernels1[i]-kernels1[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels1[i] = torch.zeros(kernels1[1]:size()) 
   end
 
   -- normalize kernels to 0 mean and 1 std:  
   kernels1[i]:add(-kernels1[i]:mean())
   kernels1[i]:div(kernels1[i]:std())
end
--print('kernels1[n]', kernels1[1])
-- visualize final kernels:
--image.display{image=kernels1:reshape(nk1,is,is), padding=2, symmetric=true, zoom=2, win=win1, nrow=math.floor(math.sqrt(nk1)), legend='1st layer filters'}

--print('==> saving centroids to disk:')
--torch.save('berkeley56x56-1l.t7', kernels1:clone())
--print('kernels1, ', #kernels1)

--[[
------------------------------------------------------------
--reconstruct the picture by using kernels
--Poor Cosmo Taking BME Course in 2013
------------------------------------------------------------
resultData = {
   data = torch.Tensor(trsize, 1, dataset.data[1][3]:size(1), dataset.data[1][3]:size(2)),
--   labels = trainData.labels:clone(),
   size = function() return trsize end
}

mlp = nn.CriterionTable(nn.MSECriterion())
intv = 1
sum=0
for i=1, trsize do
   xlua.progress(i, trsize)
   --print('kernels1[1]', kernels1[1]:reshape(is,is))
   --win4 = image.display{image=trainData.data[i][1], zoom=4, win=win4, legend='orig'} --face
   recp=torch.Tensor(trainData.data[1][1]:size(1), trainData.data[1][1]:size(2)):zero()
   for j=1, (trainData.data[1][1]:size(2)-is+1), intv do
      for k=1, (trainData.data[1][1]:size(1)-is+1), intv do
	 x=trainData.data[i][1][{{j, j+is-1},{k, k+is-1}}]
	 max = 0
	 for n=1, nk1 do
	    --print('kernels1[n]', kernels1[n]:reshape(is,is))
	    y=kernels1[n]:reshape(is,is)
	    --x:add(-x:mean())
	    --x:div(x:std())
	    out=mlp:forward{x,y}
	    if out >= max then
	       max = out
	       --ps1=j
	       --ps2=k
	       fit=y
	       --print('ok', ps1, ps2, max, 'x', 'y', x, y)
	    end
	    --print('j', j)
	 end
	 recp[{{j,j+is-1},{k,k+is-1}}]=recp[{{j,j+is-1},{k,k+is-1}}]+fit
	 --print('recp',recp[{{1,2},{3,4}}], ps1)
	 --print('recp', recp[{{ps1,ps1+is-1},{ps2,ps2+is-1}}])
      end
   end
   recp:add(-recp:mean())
   recp:div(recp:std())
   resultData.data[i]=recp
   --win5 = image.display{image=recp, zoom=4, win=win5, legend='reconp'}
   mse=mlp:forward{trainData.data[i][1], recp}
   --print('mse', mse, '\n')
   sum=sum+mse
end	  
mset=sum/trsize   
print('mset', mset)
win4 = image.display{image={trainData.data[1][1], trainData.data[2][1]}, zoom=4, win=win4, padding=2, legend='orig'} --face
win5 = image.display{image={resultData.data[1][1], resultData.data[2][1]}, zoom=4, win=win5, padding=2, legend='reconp'}
f256S = resultData.data[{{1,trsize}}]
--image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=1, legend='reconstructed images'}

]]

----------------------------------------------------------------------
print "==> loading and initialize 1st stage CL model"

o1size = trainData.data:size(3) - is + 1 -- size of spatial conv layer output
cvstepsize = 1
fanin = 1
poolsize = 2
l1netoutsize = torch.floor(o1size/poolsize/cvstepsize) -- attention, here there is a FRACTION number!

--[[
-- spatialSAD:
model = nn.Sequential()
model:add(nn.SpatialSADMap(nn.tables.random(3, nk1, fanin), is, is)) -- here all 3 input maps are = so random means nothing
model:add(nn.SpatialContrastiveNormalization(nk1, normkernel))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
model:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))

]]

-- spatial conv:
model = nn.Sequential()
model:add(nn.SpatialConvolutionMap(nn.tables.random(3, nk1, fanin), is, is))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
model:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))

l1net = model:clone()

-- initialize 1st layer parameters to learned filters (expand them for use in all channels):
l1net.modules[1].weight = kernels1:reshape(nk1, is, is)-- SpatialSAD
l1net.modules[1].bias = l1net.modules[1].bias*0

--[[
--tests:
inp = torch.Tensor(100, nk1, 28, 28)
for t = 1, 100 do
   l1net:forward(trainData.data[t])--:double())
   --print(#l1net.modules[2].output)--:max())
   --print(l1net.modules[2].output)
   inp[t] = l1net.modules[2].output
end

print('inp', #inp[10])
--print('output', inp[1])
print('MAX output after SpatialContrastNorm:', inp:mean())
image.display{image=inp[10], padding=2, symmetric=true, zoom=2, nrow=8, legend='example of 1st layer input'}
]]
--save layer:
layer1 = l1net:clone()
torch.save('berkeley-sadmap1.net', l1net.modules[1])




----------------------------------------------------------------------
print "==> processing dataset with CL network"

trainData2 = {
   data = torch.Tensor(trsize, nk1*(l1netoutsize)^2),
   labels = trainData.labels:clone(),
   size = function() return trsize end
}
for t = 1,trsize do
   trainData2.data[t] = l1net:forward(trainData.data[t])--:double())
   xlua.progress(t, trainData:size())
end

testData2 = {
   data = torch.Tensor(tssize, nk1*(l1netoutsize)^2),
   labels = testData.labels:clone(),
   size = function() return tssize end
}
for t = 1,tssize do
   testData2.data[t] = l1net:forward(testData.data[t])--:double())
   xlua.progress(t, testData:size())
end
--trainData2.data = l1net:forward(trainData.data:double())

trainData2.data = trainData2.data:reshape(trsize, nk1, l1netoutsize, l1netoutsize)
testData2.data = testData2.data:reshape(tssize, nk1, l1netoutsize, l1netoutsize)

-- relocate pointers to new dataset:
trainData1 = trainData -- save original dataset
trainData = trainData2 -- relocate new dataset

-- show a few outputs:
f256S = trainData2.data[{ {1,trsize},1 }]
--image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Output 1nd layer: first 256 examples, 1st plane'}

-- verify dataset statistics:
trainMean = trainData.data:mean()
trainStd = trainData.data:std()
print('2nd layer training data mean: ' .. trainMean)
print('2nd layer training data standard deviation: ' .. trainStd)


--------------------------------------------------------------
--torch.load('c') -- break function
--------------------------------------------------------------

----------------------------------------------------------------------
print '==> generating 2nd stage filters:'
print '==> extracting patches'
data2 = torch.Tensor(opt.nsamples,nk1*is*is)
for i = 1,opt.nsamples do
   idx = math.random(1,trainData.data:size(1))
   img = trainData.data[idx]
   z = math.random(1,trainData.data:size(2))
   x = math.random(1,trainData.data:size(3)-is+1)
   y = math.random(1,trainData.data:size(4)-is+1)
   randompatch = img[{ {},{y,y+is-1},{x,x+is-1} }]
   -- normalize patches to 0 mean and 1 std:
   randompatch:add(-randompatch:mean())
   --randompatch:div(randompatch:std())
   data2[i] = randompatch
end

-- show a few patches:
--f256S2 = data2[{{1,256}}]:reshape(256,is,is)
--image.display{image=f256S2, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches for 2nd layer learning'}

--if not paths.filep('berkeley56x56-2l.t7') then
print '==> running k-means'
function cb2 (kernels2)
   if opt.visualize then
      --win2 = image.display{image=kernels2:reshape(nk2,nk1,is,is)[{{},{1},{},{}}]:reshape(nk2,is,is), padding=2, symmetric=true, zoom=2, win=win2, nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters'} -- only one plane!
   end
end                    
kernels2 = unsup.kmeans(data2, nk2, opt.initstd2, opt.niter2, opt.batchsize, cb2, true)

-- clear nan kernels if kmeans initstd is not right!
for i=1,nk2 do   
   if torch.sum(kernels2[i]-kernels2[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels2[i] = torch.zeros(kernels2[1]:size()) 
   end
   
   -- normalize kernels to 0 mean and 1 std:  
   kernels2[i]:add(-kernels2[i]:mean())
   kernels2[i]:div(kernels2[i]:std())
end

-- visualize final kernels:
--image.display{image=kernels2:reshape(nk2,nk1,is,is)[{{},{1},{},{}}]:reshape(nk2,is,is), padding=2, symmetric=true, zoom=2, win=win2, nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters'} -- only one plane!

--print('==> saving centroids to disk:')
torch.save('berkeley56x56-2l.t7', kernels2:clone())

----------------------------------------------------------------------
print "==> initializing 2nd Stage CL model"
o1size = trainData.data:size(3) - is + 1 -- size of spatial conv layer output
--print('o1size', o1size)
fanin = 8
cvstepsize = 1
poolsize = 2
l1netoutsize = torch.floor(o1size/poolsize/cvstepsize) -- attention, here there is a FRACTION number!
--[[
-- spatialSAD:
model = nn.Sequential()
--model:add(nn.SpatialSADMap(nn.tables.random(nk1, nk2, fanin), is, is))
model:add(nn.SpatialSAD(nk1, nk2, is, is))
model:add(nn.SpatialContrastiveNormalization(nk2, normkernel))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
model:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))
]]
-- spatial conv:
model = nn.Sequential()
model:add(nn.SpatialConvolutionMap(nn.tables.random(nk1, nk2, fanin), is, is))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
model:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))
l1net = model:clone()

-- initialize 1st layer parameters to learned filters (expand them for use in all channels):
-- SpatialSAD:
--l1net.modules[1].weight = kernels2:reshape(nk2,nk1,is,is)
-- spatialSADMap:
--w2=torch.cat(kernels2:resize(nk2,is,is),kernels2:resize(nk2,is,is),1)
--w22=torch.cat(w2,w2,1)
--w222=torch.cat(w22,w22,1)
--l1net.modules[1].weight = w222
-- spatialSAD better way:
--w2=torch.cat(kernels2:resize(nk2,is,is),kernels2:resize(nk2,is,is),1)
--for t1 = 1,nk2 do
--   for t2 = t1,t1+fanin do l1net.modules[1].weight[t2] = w2[t1] end
--end

-- SpatialConv:
--l1net.modules[1].weight = kernels2:reshape(nk2,1,is,is):expand(nk2,3,is,is)--:type('torch.DoubleTensor')
-- bias:
--l1net.modules[1].bias = l1net.modules[1].bias *0

--torch.save('berkeley56x56-2l-w.t7', w222:clone())

-- random filters without k-means:
-- initialize 1st layer parameters to learned filters (expand them for use in all channels):
--l1net.modules[1].weight = kernels2:reshape(nk2, 1, is, is):expand(nk2,fanin,is,is):reshape(nk2*fanin, is, is) 

-- creating filters based on connTable:
--for i = 1, l1net.modules[1].weight:size(1) do
--   img = math.random(1,trainData.data:size(1))
--   img2 = trainData.data[img]
--   z = l1net.modules[1].connTable[i][1]
--   x = math.random(1,trainData.data:size(3)-is+1)
--   y = math.random(1,trainData.data:size(4)-is+1)
--   randompatch = img2[{ {z},{y,y+is-1},{x,x+is-1} }]
--   l1net.modules[1].weight[i] = randompatch
--end
--for i = 1, l1net.modules[1].weight:size(1) do
--   a=l1net.modules[1].connTable[i][1]
--   b=l1net.modules[1].connTable[i][2]
--   l1net.modules[1].weight[i] = kernels2:reshape(nk2,nk1,is,is)[{{b},{a},{},{}}]:reshape(is,is)
--end
-- this is for when we just use one plane in kenrnels2:
--l1net.modules[1].weight = kernels2:reshape(nk2,1,is,is):expand(nk2,fanin,is,is):reshape(nk2*fanin,is,is)

-- display filters:
--image.display{image=l1net.modules[1].weight, padding=2, symmetric=true, zoom=2, win=win2, nrow=32, legend='2nd layer filters'}

-- load kernels fully connected as done in train-cifar-CL2l-dist.lua:
--l1net.modules[1]:templates(kernels2)

--bias zeroed:
--l1net.modules[1].bias = l1net.modules[1].bias *0 -- no bias!

--tests:
--inp = torch.Tensor(100)
--for t = 1,100 do
--   l1net:forward(trainData.data[t])--:double())
--   inp[t] = l1net.modules[2].output:max()
--end
--print('MAX output after SpatialContrastNorm:', inp:mean())
--image.display{image=inp, padding=2, symmetric=true, 
--         zoom=2, nrow=8, legend='example of 1st layer output'}

--save layer:
layer2 = l1net:clone()
torch.save('berkeley-sadmap2.net', l1net.modules[1])

--------------------------------------------------------------
--torch.load('c') -- break function
--------------------------------------------------------------

----------------------------------------------------------------------
print "==> processing dataset with CL network"

trainData3 = {
   data = torch.Tensor(trsize, nk2*(l1netoutsize)^2),
   labels = trainData.labels:clone(),
   size = function() return trsize end
}
--print('trainData3.labels', #trainData3.labels[20], trainData3.labels[20])

testData3 = {
   data = torch.Tensor(tssize, nk2*(l1netoutsize)^2),
   labels = testData.labels:clone(),
   size = function() return trsize end
}

for t = 1,trsize do
   trainData3.data[t] = l1net:forward(trainData.data[t])--:double())
   trainData3.labels[t] = trainData.labels[t]
   xlua.progress(t, trainData:size())
end
--trainData2.data = l1net:forward(trainData.data:double())
--print('t3l', trainData3.labels)

for t = 1,tssize do
   testData3.data[t] = l1net:forward(testData2.data[t])--:double())
   testData3.labels[t] = testData2.labels[t]
   xlua.progress(t, testData:size())
end

trainData3.data = trainData3.data:reshape(trsize, nk2, l1netoutsize, l1netoutsize)
testData3.data = testData3.data:reshape(tssize, nk2, l1netoutsize, l1netoutsize)
--print('trainData3.data', #trainData3.data)
-- relocate pointers to new dataset:
trainData2 = trainData -- save original dataset
trainData = trainData3 -- relocate new dataset

-- show a few outputs:
f256S = trainData3.data[{{1,trsize}, 1}]
--image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Output 2nd layer: first 256 examples, 1st plane'}

-- verify dataset statistics:
trainMean = trainData.data:mean()
trainStd = trainData.data:std()
--print('3rd layer training data mean: ' .. trainMean)
--print('3rd layer training data standard deviation: ' .. trainStd)

-------------------------------------------------------------------------
--MLP Layer----
--Poor Cosmo ---
--from Dr. Culurciello--
----------------------------------------------------------------------

----------------------------------------------------------------------
print "==> creating final dataset"

--l1netoutsize = ovhe2 -- 2 layers:

trainDataF = {
   data = torch.Tensor(trsize, nk2*(l1netoutsize)^2),
   labels = trainData.labels:clone(),--torch.Tensor(trsize, 2),
			 --trainData.labels:clone(),
   size = function() return trsize end
}



testDataF = {
   data = torch.Tensor(tssize, nk2*(l1netoutsize)^2),
   labels = testData.labels:clone(), --torch.Tensor(tssize, 2),
      --testData.labels:clone(),
   size = function() return tssize end
}

for t = 1,trainDataF:size() do
   trainDataF.data[t] = trainData3.data[t]
   trainDataF.labels[t] = trainData3.labels[t]
   xlua.progress(t, trainData:size())
   --print('t', #trainData3.data[t])
   --trainDataF.data[t] = l1net:forward(trainData3.data[t])--:double())
   --xlua.progress(t, trainData:size())
end

for t = 1,testDataF:size() do
   testDataF.data[t] = testData3.data[t]
   testDataF.labels[t] = testData3.labels[t]
   --testDataF.data[t] = l1net:forward(testData3.data[t])--:double())
   xlua.progress(t, testData:size())
end

--print(#trainDataF) --printtest

trainDataF.data = trainDataF.data:reshape(trsize, nk2, l1netoutsize, l1netoutsize)
testDataF.data = testDataF.data:reshape(tssize, nk2, l1netoutsize, l1netoutsize)

--print('tdfl', trainDataF.labels) 
-- relocate pointers to new dataset:
--trainData1 = trainData -- save original dataset
--testData1 = testData
trainData = trainDataF -- relocate new dataset
testData = testDataF

--print('tdfl', trainData.labels) 
--print('tb', trainData.labels[1])

-------------------------------------------------------------------------
print "==> initializing 3rd Stage layer MLP model"
----------------------------------------------------------------------
-- classifier for train/test:
----------------------------------------------------------------------
print "==> creating classifier"

--   opt.model = '2mlp-classifier'
--   dofile '2_model.lua' 

nhiddens = 256
outsize = nclass -- in CIFAR, SVHN datasets

model = nn.Sequential()
model:add(nn.Reshape(nk2*l1netoutsize^2))
model:add(nn.Linear(nk2*l1netoutsize^2, nhiddens))
model:add(nn.Threshold())
model:add(nn.Linear(nhiddens, outsize))

print "==> test network output:"
print(model:forward(trainData.data[1]))--:double()))

dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print "==> training classifier"

while true do
   train()
   test()
end

