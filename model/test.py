def pdb_2_mrc(file_name,apix=1.0,res=2.8,het=False,box=None,chains=None,model=None,center=False,quiet=False):
	'''
	file_name is the name of a pdb file
	apix is the angstrom per pixel
	res is requested resolution, equivalent to Gaussian lowpass with 1/e width at 1/res
	het is a flag indicating whether HET atoms should be included in the map
	box is the boxsize, can be a single int (e.g. 128), a tuple (e.g. [128,64,54]), or a string (e.g. "128" or "128,64,57")
	chains is a string list of chain identifiers, eg 'ABEFG'
	quiet can be used to turn of helpful print outs
	'''

	try : infile=open(file_name,"r")
	except : raise IOError("%s is an invalid file name" %file_name)

	if res<=apix : print("Warning: res<=apix. Generally res should be 2x apix or more")

	aavg=[0,0,0]	# calculate atomic center
	amin=[1.0e20,1.0e20,1.0e20]		# coords of lower left front corner of bounding box
	amax=[-1.0e20,-1.0e20,-1.0e20]	# coords
	natm=0
	atoms=[]		# we store a list of atoms to process to avoid multiple passes
	nelec=0
	mass=0

	# parse the pdb file and pull out relevant atoms
	stm=False
	for line in infile:
		if model!=None:
			if line[:5]=="MODEL":
				if int(line.split()[1])==model: stm=True
			if stm and line[:6]=="ENDMDL" : break
			if not stm: continue

		if line[:4]=='ATOM' or (line[:6]=='HETATM' and het) :
			if chains and not (line[21] in chains) : continue

			try:
				aseq=int(line[6:11].strip())
			except:
				aseq=-1

			try:
				resol=int(line[22:26].strip())
			except:
				resol=10

			try:
				a=line[12:14].strip().translate(transmap)

				x=float(line[30:38])
				y=float(line[38:46])
				z=float(line[46:54])
			except:
				print("PDB Parse error:\n%s\n'%s','%s','%s'  '%s','%s','%s'\n"%(
					line,line[12:14],line[6:11],line[22:26],line[30:38],line[38:46],line[46:54]))
				print(a,aseq,resol,x,y,z)

			atoms.append((a,x,y,z))

			aavg[0]+=x
			aavg[1]+=y
			aavg[2]+=z
			natm+=1

			amin[0]=min(x,amin[0])
			amin[1]=min(y,amin[1])
			amin[2]=min(z,amin[2])
			amax[0]=max(x,amax[0])
			amax[1]=max(y,amax[1])
			amax[2]=max(z,amax[2])

			try:
				nelec+=atomdefs[a.upper()][0]
				mass+=atomdefs[a.upper()][1]
			except:
				print(("Unknown atom %s ignored at %d"%(a,aseq)))

	infile.close()

	if not quiet:
		print("%d atoms used with a total charge of %d e- and a mass of %d kDa"%(natm,nelec,old_div(mass,1000)))
		print("atomic center at %1.1f,%1.1f,%1.1f (center of volume at 0,0,0)"%(old_div(aavg[0],natm),old_div(aavg[1],natm),old_div(aavg[2],natm)))
		print("Bounding box: x: %7.2f - %7.2f"%(amin[0],amax[0]))
		print("              y: %7.2f - %7.2f"%(amin[1],amax[1]))
		print("              z: %7.2f - %7.2f"%(amin[2],amax[2]))

	# precalculate a prototypical Gaussian to resample
	# 64^3 box with a real-space 1/2 width of 12 pixels
	gaus=EMData()
	gaus.set_size(64,64,64)
	gaus.to_one()

	gaus.process_inplace("mask.gaussian",{"outer_radius":12.0})

	# find the output box size, either user specified or from bounding box
	outbox=[0,0,0]
	try:
		# made
		if isinstance(box,int):
			outbox[0]=outbox[1]=outbox[2]=box
		elif isinstance(box,list):
			outbox[0]=box[0]
			outbox[1]=box[1]
			outbox[2]=box[2]
		else:
			spl=box.split(',')
			if len(spl)==1 : outbox[0]=outbox[1]=outbox[2]=int(spl[0])
			else :
				outbox[0]=int(spl[0])
				outbox[1]=int(spl[1])
				outbox[2]=int(spl[2])
	except:
		pad=int(2.0*res/apix)
		outbox[0]=int(old_div((amax[0]-amin[0]),apix))+pad
		outbox[1]=int(old_div((amax[1]-amin[1]),apix))+pad
		outbox[2]=int(old_div((amax[2]-amin[2]),apix))+pad
		outbox[0]+=outbox[0]%2
		outbox[1]+=outbox[1]%2
		outbox[2]+=outbox[2]%2

	if not quiet: print("Box size: %d x %d x %d"%(outbox[0],outbox[1],outbox[2]))

	# initialize the final output volume
	outmap=EMData()
	outmap.set_size(outbox[0],outbox[1],outbox[2])
	outmap.to_zero()
	for i in range(len(aavg)): aavg[i] = old_div(aavg[i],float(natm))
	# fill in the atom gaussians
	xt = old_div(outbox[0],2) - old_div((amax[0]-amin[0]),(2*apix))
	yt = old_div(outbox[1],2) - old_div((amax[1]-amin[1]),(2*apix))
	zt = old_div(outbox[2],2) - old_div((amax[2]-amin[2]),(2*apix))
	for i,a in enumerate(atoms):
		if not quiet and i%1000==0 :
			print('\r   %d'%i, end=' ')
			sys.stdout.flush()
		try:
			# This insertion strategy ensures the output is centered.
			elec=atomdefs[a[0].translate(transmap).upper()][0]
			# This was producing different results than the "quick" mode, and did not match the statement printed above!!!
			#outmap.insert_scaled_sum(gaus,(a[1]/apix+xt-amin[0]/apix,a[2]/apix+yt-amin[1]/apix,a[3]/apix+zt-amin[2]/apix),res/(pi*12.0*apix),elec)
			if center: outmap.insert_scaled_sum(gaus,(old_div((a[1]-aavg[0]),apix)+old_div(outbox[0],2),old_div((a[2]-aavg[1]),apix)+old_div(outbox[1],2),old_div((a[3]-aavg[2]),apix)+old_div(outbox[2],2)),old_div(res,(pi*12.0*apix)),elec)
			else: outmap.insert_scaled_sum(gaus,(old_div(a[1],apix)+old_div(outbox[0],2),old_div(a[2],apix)+old_div(outbox[1],2),old_div(a[3],apix)+old_div(outbox[2],2)),old_div(res,(pi*12.0*apix)),elec)
		except: print("Skipping %d '%s'"%(i,a[0]))
	if not quiet: print('\r   %d\nConversion complete'%len(atoms))
	outmap.set_attr("apix_x",apix)
	outmap.set_attr("apix_y",apix)
	outmap.set_attr("apix_z",apix)
	outmap.set_attr("origin_x",-xt*apix+amin[0])
	outmap.set_attr("origin_y",-yt*apix+amin[1])
	outmap.set_attr("origin_z",-zt*apix+amin[2])
	return outmap