root_folder = '/share/project/xiaotongli/tta/data/imagenet-adversarial/imagenet-a'  # 输入的文件夹
intermediate_folder = 'intermediate'  # 存放中间文件的文件夹
new_folder = '/share/project/xiaotongli/tta/data/imagenet-adversarial/new'  # 输出解压后图片的文件夹

def compress_decompress(args):
    """Compresses and Decompress images."""
    global device
    mode = 'low'
    if args.qp > 3:
        mode = 'high'
    device = torch.device(args.device)
    qp = args.qp
    model_type = args.model_type
    print(f'model_type: {model_type}, qp: {qp}')
    if mode == 'low':
        net = NetLow().eval()
    else:
        net = NetHigh().eval()
    net = net.to(device)
    # sd = net.state_dict()
    # td = load_tf_weights(sd, f'models/model{model_type}_qp{qp}.pk')
    # Changed to directly load the weights
    # Here we only manually load one fixed model for testing
    # You can write some code to read from command line arguments
    td = torch.load('trained_model.ckpt', map_location=torch.device(device))
    ntd = {}
    # naive way to remove 'module.' before the name of each tensor
    for k in td:
        if k[:6] == 'module':
            ntd[k[7:]] = td[k]
        else:
            ntd[k] = td[k]
    ret = net.load_state_dict(ntd, strict=False)
    print(ret)

    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(new_folder, exist_ok=True)

    # 遍历根文件夹中的所有子文件夹和图片文件
    # 统计所有图片文件的总数
    all_images = []

    # 遍历根文件夹中的所有子文件夹和图片文件
    for subdir, _, files in os.walk(root_folder):
        # 只处理直接在 root_folder 下的文件和子文件夹中的文件
        if subdir == root_folder:
            # 根目录下的文件直接处理
            files_to_process = files
            relative_subdir = ''
        else:
            # 子目录中的文件处理
            files_to_process = files
            relative_subdir = os.path.relpath(subdir, root_folder)

        for file in files_to_process:
            # 检查文件是否是图片（jpg或png格式）
            if file.lower().endswith(('.jpg', '.png', 'jpeg')):
                all_images.append((subdir, file, relative_subdir))

    # 创建一个进度条来显示所有文件处理的进度
    with tqdm(total=len(all_images), desc="Processing Images", unit="file") as pbar:
        for subdir, file, relative_subdir in all_images:
            old_path = os.path.join(subdir, file)
            # 按照子文件夹名称来组织新文件夹结构
            if relative_subdir:
                new_subdir = os.path.join(new_folder, relative_subdir)
            else:
                new_subdir = new_folder
            os.makedirs(new_subdir, exist_ok=True)
            new_path = os.path.join(new_subdir, file)

            # ==================================================压缩文件=================================================
            if relative_subdir:
                intermediate_path = os.path.join(intermediate_folder, relative_subdir, file.split('.')[0] + '.bin')
            else:
                intermediate_path = os.path.join(intermediate_folder, file.split('.')[0] + '.bin')
            # Load input image and add batch dimension.
            f = Image.open(old_path)
            fshape = [f.size[1], f.size[0], 3]
            x = np.array(f).reshape([1, ] + fshape)

            fileobj = open(intermediate_path, mode='wb')

            buf = qp << 1
            buf = buf + model_type
            arr = np.array([0], dtype=np.uint8)
            arr[0] = buf
            arr.tofile(fileobj)

            h, w = (fshape[0] // 64) * 64, (fshape[1] // 64) * 64
            if h < fshape[0]:
                h += 64
            if w < fshape[1]:
                w += 64

            pad_up = (h - fshape[0]) // 2
            pad_down = (h - fshape[0]) - pad_up
            pad_left = (w - fshape[1]) // 2
            pad_right = (w - fshape[1]) - pad_left

            x = np.pad(x, [[0, 0], [pad_up, pad_down], [pad_left, pad_right], [0, 0]])
            torch_x = (torch.tensor(x).permute(0, 3, 1, 2).float() / 127.5) - 1

            arr = np.array([fshape[0], fshape[1]], dtype=np.uint16)
            arr.tofile(fileobj)
            fileobj.close()

            hs, ws = get_partition(h, w)
            blocks = []

            ph = 0
            for th in hs:
                pw = 0
                for tw in ws:
                    blocks.append(torch_x[:, :, ph:ph + th, pw:pw + tw])
                    pw += tw
                ph += th

            cnt = 0
            for block in blocks:
                with torch.no_grad():
                    block = block.to(device)
                    ret = net.encode(block)
                    bsf = open(intermediate_path, "ab+")
                    v_z1_sigma = ret['z1_sigma']
                    v_z1_rounded = ret['z1_rounded']

                    sigma_val = np.reshape(v_z1_sigma, (1, -1, 1, 1))
                    tshape = v_z1_rounded.shape
                    sigma_val = np.tile(sigma_val, (tshape[0], 1, tshape[2], tshape[3]))
                    mu_val = np.zeros_like(sigma_val)
                    mu_val[...] = 511
                    flat_sigma = sigma_val.reshape((-1))
                    flat_mu = mu_val.reshape((-1))
                    flat_coeff = (v_z1_rounded + 511).astype(np.int16).reshape((-1))
                    length = np.array([flat_sigma.shape[0]], dtype=np.int64)
                    s = length.tobytes() + flat_coeff.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes()
                    r = sp.run([EXE_ARITH, 'e', f'{mask_a}', f'{mask_b}'], input=s, stdout=sp.PIPE)
                    bslen = np.array([len(r.stdout)], dtype=np.int64)
                    bsf.write(bslen.tobytes())
                    masks = np.array([mask_a, mask_b], dtype=np.uint8)
                    bsf.write(masks.tobytes())
                    bsf.write(r.stdout)
                    # print(length, bslen)

                    v_z2_sigma = ret['z2_sigma']
                    v_z2_mu = ret['z2_mu']
                    v_z2_rounded = ret['z2_rounded']

                    sigma_val = v_z2_sigma
                    tshape = v_z2_rounded.shape
                    mu_val = v_z2_mu + 511
                    flat_sigma = sigma_val.reshape((-1))
                    flat_mu = mu_val.reshape((-1))
                    flat_coeff = (v_z2_rounded + 511).astype(np.int16).reshape((-1))
                    length = np.array([flat_sigma.shape[0]], dtype=np.int64)
                    s = length.tobytes() + flat_coeff.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes()
                    r = sp.run([EXE_ARITH, 'e', f'{mask_a}', f'{mask_b}'], input=s, stdout=sp.PIPE)
                    bslen = np.array([len(r.stdout)], dtype=np.int64)
                    bsf.write(bslen.tobytes())
                    masks = np.array([mask_a, mask_b], dtype=np.uint8)
                    bsf.write(masks.tobytes())
                    bsf.write(r.stdout)
                    # print(length, bslen)

                    v_z3_sigma = ret['z3_sigma']
                    v_z3_mu = ret['z3_mu']
                    v_z3_rounded = ret['z3_rounded']

                    sigma_val = v_z3_sigma
                    tshape = v_z3_rounded.shape
                    mu_val = v_z3_mu + 511
                    flat_sigma = sigma_val.reshape((-1))
                    flat_mu = mu_val.reshape((-1))
                    flat_coeff = (v_z3_rounded + 511).astype(np.int16).reshape((-1))
                    length = np.array([flat_sigma.shape[0]], dtype=np.int64)
                    s = length.tobytes() + flat_coeff.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes()
                    r = sp.run([EXE_ARITH, 'e', f'{mask_a}', f'{mask_b}'], input=s, stdout=sp.PIPE)
                    bslen = np.array([len(r.stdout)], dtype=np.int64)
                    bsf.write(bslen.tobytes())
                    masks = np.array([mask_a, mask_b], dtype=np.uint8)
                    bsf.write(masks.tobytes())
                    bsf.write(r.stdout)
                    # print(length, bslen)

                    bsf.close()
                    cnt += 1
            # ============================================解压缩文件======================================================
            fileobj = open(intermediate_path, mode='rb')
            buf = fileobj.read(1)
            arr = np.frombuffer(buf, dtype=np.uint8)
            model_type = arr[0] % 2
            qp = arr[0] >> 1
            mode = 'low'
            if qp > 3:
                mode = 'high'
            print(f'model_type: {model_type}, qp: {qp}')

            buf = fileobj.read(4)
            arr = np.frombuffer(buf, dtype=np.uint16)
            h = int(arr[0])
            w = int(arr[1])
            oh = h
            ow = w

            fshape = [h, w]

            h, w = (fshape[0] // 64) * 64, (fshape[1] // 64) * 64
            if h < fshape[0]:
                h += 64
            if w < fshape[1]:
                w += 64

            pad_up = (h - fshape[0]) // 2
            pad_down = (h - fshape[0]) - pad_up
            pad_left = (w - fshape[1]) // 2
            pad_right = (w - fshape[1]) - pad_left

            padded_w = int(math.ceil(w / 16) * 16)
            padded_h = int(math.ceil(h / 16) * 16)

            def decode_block(padded_h, padded_w):
                with torch.no_grad():
                    z1_hat = np.zeros((1, 32 * 4, padded_h // 64, padded_w // 64), dtype=np.float32)
                    z2_hat = np.zeros((1, 64 * 4, padded_h // 32, padded_w // 32), dtype=np.float32)
                    if mode == 'low':
                        z3_hat = np.zeros((1, 192, padded_h // 16, padded_w // 16), dtype=np.float32)
                    else:
                        z3_hat = np.zeros((1, 384, padded_h // 16, padded_w // 16), dtype=np.float32)

                    ret = net.decode(None, 0)
                    sigma_z = ret['z1_sigma']

                    bslenbs = fileobj.read(8)
                    bslen = np.frombuffer(bslenbs, dtype=np.int64)
                    maskbs = fileobj.read(2)
                    masks = np.frombuffer(maskbs, dtype=np.uint8)
                    bs = fileobj.read(bslen[0])
                    length = np.array(z1_hat.reshape((-1)).shape[0], dtype=np.int64)
                    sigma_val = np.reshape(sigma_z, (1, -1, 1, 1))
                    tshape = z1_hat.shape
                    sigma_val = np.tile(sigma_val, (tshape[0], 1, tshape[2], tshape[3]))
                    mu_val = np.zeros_like(sigma_val)
                    mu_val[...] = 511
                    flat_sigma = sigma_val.reshape((-1))
                    flat_mu = mu_val.reshape((-1))
                    s = length.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes() + bs
                    r = sp.run(['./module_arithmeticcoding', 'd', f'{masks[0]}', f'{masks[1]}'], input=s,
                               stdout=sp.PIPE)
                    coeffs = np.frombuffer(r.stdout, dtype=np.int16)
                    z1_hat[...] = coeffs.reshape(z1_hat.shape).astype(np.float32) - 511

                    s2_in = {'z1_rounded': torch.tensor(z1_hat).to(device)}

                    ret = net.decode(s2_in, 1)
                    v_z2_sigma = ret['z2_sigma']
                    v_z2_mu = ret['z2_mu']

                    bslenbs = fileobj.read(8)
                    bslen = np.frombuffer(bslenbs, dtype=np.int64)
                    maskbs = fileobj.read(2)
                    masks = np.frombuffer(maskbs, dtype=np.uint8)
                    bs = fileobj.read(bslen[0])
                    length = np.array(z2_hat.reshape((-1)).shape[0], dtype=np.int64)
                    sigma_val = v_z2_sigma
                    tshape = z2_hat.shape
                    mu_val = v_z2_mu + 511
                    flat_sigma = sigma_val.reshape((-1))
                    flat_mu = mu_val.reshape((-1))
                    s = length.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes() + bs
                    r = sp.run(['./module_arithmeticcoding', 'd', f'{masks[0]}', f'{masks[1]}'], input=s,
                               stdout=sp.PIPE)
                    coeffs = np.frombuffer(r.stdout, dtype=np.int16)
                    z2_hat[...] = coeffs.reshape(z2_hat.shape).astype(np.float32) - 511

                    s3_in = {'z2_rounded': torch.tensor(z2_hat).to(device)}
                    ret = net.decode(s3_in, 2)
                    v_z3_sigma = ret['z3_sigma']
                    v_z3_mu = ret['z3_mu']

                    bslenbs = fileobj.read(8)
                    bslen = np.frombuffer(bslenbs, dtype=np.int64)
                    maskbs = fileobj.read(2)
                    masks = np.frombuffer(maskbs, dtype=np.uint8)
                    bs = fileobj.read(bslen[0])
                    length = np.array(z3_hat.reshape((-1)).shape[0], dtype=np.int64)
                    sigma_val = v_z3_sigma
                    tshape = z3_hat.shape
                    mu_val = v_z3_mu + 511
                    flat_sigma = sigma_val.reshape((-1))
                    flat_mu = mu_val.reshape((-1))
                    s = length.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes() + bs
                    r = sp.run(['./module_arithmeticcoding', 'd', f'{masks[0]}', f'{masks[1]}'], input=s,
                               stdout=sp.PIPE)
                    coeffs = np.frombuffer(r.stdout, dtype=np.int16)
                    z3_hat[...] = coeffs.reshape(z3_hat.shape).astype(np.float32) - 511

                    s4_in = {'z3_rounded': torch.tensor(z3_hat).to(device)}
                    x_tilde = net.decode(s4_in, 3)
                    return x_tilde

            hs, ws = get_partition(padded_h, padded_w)
            blocks = []

            tar = np.zeros((1, 3, padded_h, padded_w), dtype=np.float32)
            ph = 0
            for th in hs:
                pw = 0
                for tw in ws:
                    tar[:, :, ph:ph + th, pw:pw + tw] = decode_block(th, tw)
                    pw += tw
                ph += th

            if qp < 3:
                ph = 0
                for th in hs[:-1]:
                    ph += th
                    blks = []
                    for x in range(0, padded_w - 8, 4):
                        blk = tar[0, :, ph - 4:ph + 4, x:x + 8]
                        im = blk.astype(np.uint8).transpose((1, 2, 0))
                        img = Image.fromarray(im)
                        img = img.filter(ImageFilter.SMOOTH)
                        imt = np.array(img).transpose((2, 0, 1)).astype(np.float32)
                        blks.append(imt)
                    cnt = 0
                    for x in range(0, padded_w - 8, 4):
                        tar[0, :, ph - 2:ph + 2, x + 2:x + 6] = blks[cnt][:, 2:-2, 2:-2]
                        cnt += 1

                pw = 0
                for tw in ws[:-1]:
                    pw += tw
                    blks = []
                    for y in range(0, padded_h - 8, 4):
                        blk = tar[0, :, y:y + 8, pw - 4:pw + 4]
                        im = blk.astype(np.uint8).transpose((1, 2, 0))
                        img = Image.fromarray(im)
                        img = img.filter(ImageFilter.SMOOTH)
                        imt = np.array(img).transpose((2, 0, 1)).astype(np.float32)
                        blks.append(imt)
                    cnt = 0
                    for y in range(0, padded_h - 8, 4):
                        tar[0, :, y + 2:y + 6, pw - 2:pw + 2] = blks[cnt][:, 2:-2, 2:-2]
                        cnt += 1

            def make_img(x_tilde, fn):
                im = x_tilde.astype(np.uint8).transpose((1, 2, 0))
                img = Image.fromarray(im)
                img.save(fn)

            if pad_down != 0 and pad_right != 0:
                make_img(tar[0, :, pad_up:-pad_down, pad_left:-pad_right], new_path)
            elif pad_down != 0:
                make_img(tar[0, :, pad_up:-pad_down, pad_left:], new_path)
            elif pad_right != 0:
                make_img(tar[0, :, pad_up:, pad_left:-pad_right], new_path)
            else:
                make_img(tar[0, :, pad_up:, pad_left:], new_path)

            # 更新进度条
            pbar.update(1)
