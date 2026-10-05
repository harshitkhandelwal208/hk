import re,sys
src=open('vk_all.zig').read()
# split into top-level declarations: lines starting with "pub const|pub extern|pub fn|pub var|const"
lines=src.split('\n')
decls=[]  # (name,text)
cur=[]; depth=0
def flush():
    global cur
    if cur:
        text='\n'.join(cur)
        m=re.match(r'(?:pub )?(?:const|var|extern fn|fn|extern "c" fn)\s+(@?"?[A-Za-z_0-9]+"?)',text)
        name=m.group(1) if m else None
        decls.append((name,text))
    cur=[]
for ln in lines:
    if depth==0 and (ln.startswith('pub ') or ln.startswith('const ') or ln.startswith('extern ')) and cur:
        flush()
    cur.append(ln)
    depth+=ln.count('{')-ln.count('}')
    depth+=ln.count('(')-ln.count(')')
flush()
import re as _re
decls=[(n,_re.sub(r'(?m)^\s+pub const vk\w+ = __root\.vk\w+;\n?','',t)) for n,t in decls]
byname={n:t for n,t in decls if n}
order=[n for n,t in decls if n]
print(len(order),'decls',file=sys.stderr)

roots=[l.strip() for l in open('roots.txt') if l.strip() and not l.startswith('#')]
need=set(); stack=list(roots)
ident=re.compile(r'[A-Za-z_][A-Za-z_0-9]*')
missing=[]
while stack:
    n=stack.pop()
    if n in need: continue
    if n not in byname:
        missing.append(n); continue
    need.add(n)
    for m in ident.findall(byname[n]):
        if m in byname and m not in need: stack.append(m)
print('missing',missing,file=sys.stderr)
out=[ '//! Vulkan declarations used by hk, extracted from the Khronos Vulkan-Headers (vulkan_core.h,',
 '//! version 1.4.365, Apache-2.0 OR MIT) by `zig translate-c`. Only what the GPU backend uses is kept.',
 '//! Regenerate with tools/gen_vk_bindings.py if more of the API is needed.','']
for n in order:
    if n in need: out.append(byname[n])
open('vk_min.zig','w').write('\n'.join(out)+'\n')
print(len(need),'kept',file=sys.stderr)
