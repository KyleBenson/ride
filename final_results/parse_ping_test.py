import sys

# ignore any 'gaps' in responses with fewer than this many
MIN_GAP_SIZE = 9
MS_PER_PING = 200  # for calculating gap length in seconds

try:
    fname = sys.argv[1]
except IndexError:
    raise SystemExit("must specify ping output filename to parse!")

with open(fname) as f:
    lines = f.readlines()

# IDEA: Parse each line individually by examining whether it's a successful response or not.
# we'll record the responses as a dict with the key being seq# (every expected seq# will be present!)
# and the value being whether it was successful or not
# NOTE: since ping doesn't always start with seq# 1, let's add a dummy 0...
responses = {0: True}

def record_response(seqn, status):
    # double-check we haven't seen this one already!
    if seqn in responses:
        print "Found duplicate! %d already present in collection; was %s now %s" % (seqn, responses[seqn], status)
    responses[seqn] = status

def record_explicit_failure(seqn):
    """seqn is the next seq# we expected to see and we've already verified that it was not a successful echo response"""
    record_response(seqn, False)

def record_success(seqn):
    record_response(seqn, True)

def record_implicit_failures(seqn, last_seen):
    """last_seen is the last seq# we saw, whether successful or not, so we need to record this verified implicit gap
    by recording each missing seq#"""

    # since we're dealing with long ints, we can't just use range...
    i = last_seen + 1
    while i < seqn:
        record_explicit_failure(i)
        i += 1

def parse_seq_num(seqn):
    """parses the seqn from a string into a long integer, where if it wraps around we keep it increasing using longs!"""
    num_wraparounds = len(responses) / 65536
    # print 'nwraps:', num_wraparounds
    parsed = long(seqn) + (long(65536) * num_wraparounds)
    # print 'parsed seq from %s: %ld' % (seqn, parsed)
    return parsed

# to see if we totally missed responses somehow (e.g. timeout that wasn't printed to file?)
last_seqn = long(0)

del lines[0]  # skip over first line since it's just saying that we're starting ping
for l in lines:
    if 'bytes from' in l:
        seqn = parse_seq_num(l.split()[5].split('=')[1])
        # last seq# we saw is not immediately followed by the current one: implicit gap!
        if seqn != last_seqn + 1 and last_seqn > -1:
            record_implicit_failures(seqn, last_seqn)

        record_success(seqn)
        # print 'seq#:', seqn
    elif 'Destination Host Unreachable' in l:
        seqn = parse_seq_num(l.split()[3].split('=')[1])
        if seqn != last_seqn + 1:
            record_implicit_failures(seqn, last_seqn)

        record_explicit_failure(seqn)
        # print "HOST UNREACHABLE! seq#:", seqn
    else:
        print "Unexpected line:", l
    last_seqn = seqn

# ensure that the gaps have complete sequentially-ordered seq#s where each subsequent seq# is 1 greater than the previous,
# UNLESS we've looped back around to seq#=0
sorted_seq_nums = sorted(responses.keys())
last_seq = sorted_seq_nums[0]
for next_seq in sorted_seq_nums[1:]:
    # looped back around!
    if next_seq == 0:
        assert last_seq == 65535
    else:
        assert last_seq + 1 == next_seq
    last_seq = next_seq

# Now, convert the sequence of responses to 'gaps', where each gap is a range of the seq #s without responses.
# NOTE: all we really care about is start/end...
gaps = []

for seqn in sorted_seq_nums:
    status = responses[seqn]
    if status:
        continue

    last_gap = gaps[-1] if len(gaps) > 0 else []
    last_seq = last_gap[-1] if len(last_gap) > 0 else long(-1)

    # part of the last gap?
    if seqn == last_seq + 1:
        last_gap.append(seqn)
    # or new one?
    else:
        gaps.append([seqn])

def seq_to_seconds(seqn):
    return seqn/(1000.0/MS_PER_PING)

# print out gaps as their lengths and ranges, skipping very small ones and converting gap length to seconds
gap_str = ["%ds down-time starting after %fmins with seq# range: [%d --> %d]" % (seq_to_seconds(len(g)), seq_to_seconds(g[0])/60, g[0], g[-1]) for g in gaps if len(g) >= MIN_GAP_SIZE]
gap_str = "\n".join(gap_str)
print "\nGAPS ENCOUNTERED:\n", gap_str